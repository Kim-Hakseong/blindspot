"""Blindspot on AWS.

    CLI --StartExecution--> Step Functions: Plan -> SubmitWave (Batch array) -> Plan ...
                                              |-> Finalize (envelope to S3)
                                              |-> Halt (AWAITING_APPROVAL)
    Batch on Fargate, arm64 (Graviton) and x86-64 queues, same worker image
    DynamoDB ledger: bs-runs / bs-probes / bs-decisions     S3: datasets, waves, envelopes

Cost shape, because the account is paid for personally: public subnets only
(no NAT gateway, no interface endpoints), on-demand tables, Fargate capped at
MAX_VCPUS per compute environment, one-week logs, and a Budgets alarm filtered
to this project's tag. Every resource is tagged project=blindspot and named
bs-*, because the account also hosts an unrelated project.
"""

from __future__ import annotations

import pathlib

import aws_cdk as cdk
from aws_cdk import (
    aws_batch as batch,
    aws_budgets as budgets,
    aws_dynamodb as ddb,
    aws_ec2 as ec2,
    aws_ecr_assets as ecr_assets,
    aws_ecs as ecs,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_logs as logs,
    aws_s3 as s3,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
)
from constructs import Construct

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAX_VCPUS = 16
WORKER_CPU = 2
WORKER_MEMORY_MIB = 4096
BUDGET_USD = 25


class BlindspotStack(cdk.Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        cdk.Tags.of(self).add("project", "blindspot")
        skip_docker = self.node.try_get_context("skip_docker") == "1"

        # ---- network: public subnets only, so no NAT gateway to pay for ----
        vpc = ec2.Vpc(
            self, "Vpc", vpc_name="bs-vpc", max_azs=2, nat_gateways=0,
            subnet_configuration=[ec2.SubnetConfiguration(
                name="public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24)],
        )
        # S3 gateway endpoints are free and keep dataset traffic off the internet.
        vpc.add_gateway_endpoint("S3", service=ec2.GatewayVpcEndpointAwsService.S3)
        sg = ec2.SecurityGroup(self, "WorkerSg", vpc=vpc, security_group_name="bs-worker",
                               description="Blindspot probe workers: egress only",
                               allow_all_outbound=True)

        # ---- state ----
        bucket = s3.Bucket(
            self, "Artifacts",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True, encryption=s3.BucketEncryption.S3_MANAGED,
            # No auto_delete_objects: its helper role and Lambda are created
            # outside the tag aspect, i.e. untagged resources in a shared
            # account. tools/teardown.sh empties the bucket before destroy.
            removal_policy=cdk.RemovalPolicy.DESTROY,
            lifecycle_rules=[s3.LifecycleRule(prefix="runs/", expiration=cdk.Duration.days(30))],
        )

        def table(name, sort_key=None):
            return ddb.Table(
                self, name.replace("-", "_"), table_name=name,
                partition_key=ddb.Attribute(name="run_id", type=ddb.AttributeType.STRING),
                sort_key=ddb.Attribute(name=sort_key, type=ddb.AttributeType.STRING) if sort_key else None,
                billing_mode=ddb.BillingMode.PAY_PER_REQUEST,
                removal_policy=cdk.RemovalPolicy.DESTROY,
            )

        runs = table("bs-runs")
        probes = table("bs-probes", "probe_id")
        decisions = table("bs-decisions", "decision_id")

        # ---- worker image, both architectures (rule K1) ----
        worker_log = logs.LogGroup(self, "WorkerLogs", log_group_name="/blindspot/worker",
                                   retention=logs.RetentionDays.ONE_WEEK,
                                   removal_policy=cdk.RemovalPolicy.DESTROY)
        job_role = iam.Role(self, "WorkerRole", role_name="bs-worker-role",
                            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"))
        bucket.grant_read_write(job_role)
        probes.grant_write_data(job_role)

        queues, job_defs = {}, {}
        for arch, cpu_arch, platform in (
            ("arm64", ecs.CpuArchitecture.ARM64, ecr_assets.Platform.LINUX_ARM64),
            ("x86", ecs.CpuArchitecture.X86_64, ecr_assets.Platform.LINUX_AMD64),
        ):
            image = (ecs.ContainerImage.from_registry("public.ecr.aws/docker/library/busybox:1.36")
                     if skip_docker else
                     ecs.ContainerImage.from_docker_image_asset(ecr_assets.DockerImageAsset(
                         self, f"WorkerImage_{arch}", directory=str(ROOT), platform=platform,
                         file="Dockerfile",
                         exclude=["viewer", "infra", ".venv", ".cache", "val/*/images",
                                  "bench/out/hook_cells", "node_modules", ".git"])))
            env = batch.FargateComputeEnvironment(
                self, f"Compute_{arch}", compute_environment_name=f"bs-{arch}",
                vpc=vpc, vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
                security_groups=[sg], maxv_cpus=MAX_VCPUS, spot=False,
            )
            queues[arch] = batch.JobQueue(self, f"Queue_{arch}", job_queue_name=f"bs-{arch}",
                                          compute_environments=[batch.OrderedComputeEnvironment(
                                              compute_environment=env, order=1)])
            job_defs[arch] = batch.EcsJobDefinition(
                self, f"Worker_{arch}", job_definition_name=f"bs-worker-{arch}",
                timeout=cdk.Duration.minutes(30), retry_attempts=2,
                container=batch.EcsFargateContainerDefinition(
                    self, f"WorkerContainer_{arch}", image=image,
                    cpu=WORKER_CPU, memory=cdk.Size.mebibytes(WORKER_MEMORY_MIB),
                    fargate_cpu_architecture=cpu_arch,
                    fargate_operating_system_family=ecs.OperatingSystemFamily.LINUX,
                    assign_public_ip=True, job_role=job_role,
                    logging=ecs.LogDriver.aws_logs(stream_prefix=arch, log_group=worker_log),
                    command=["worker", "--spec", "Ref::spec", "--table", probes.table_name],
                    environment={"BLINDSPOT_BUCKET": bucket.bucket_name,
                                 "BLINDSPOT_ARCH": arch},
                ),
                parameters={"spec": "unset"},
            )

        # ---- planner: pure plan_round behind a thin adapter ----
        plan_log = logs.LogGroup(self, "PlanLogs", log_group_name="/blindspot/plan",
                                 retention=logs.RetentionDays.ONE_WEEK,
                                 removal_policy=cdk.RemovalPolicy.DESTROY)
        plan_fn = lambda_.Function(
            self, "PlanFn", function_name="bs-plan",
            runtime=lambda_.Runtime.PYTHON_3_12, architecture=lambda_.Architecture.ARM_64,
            handler="blindspot.cloud.plan_handler.handler",
            code=lambda_.Code.from_asset(str(ROOT / "src"), exclude=["**/__pycache__"]),
            timeout=cdk.Duration.seconds(60), memory_size=256, log_group=plan_log,
            environment={"RUNS_TABLE": runs.table_name, "PROBES_TABLE": probes.table_name,
                         "DECISIONS_TABLE": decisions.table_name,
                         "BLINDSPOT_BUCKET": bucket.bucket_name},
        )
        runs.grant_read_write_data(plan_fn)
        probes.grant_read_data(plan_fn)
        decisions.grant_write_data(plan_fn)
        bucket.grant_read_write(plan_fn)

        def invoke(name, mode):
            return tasks.LambdaInvoke(
                self, name, lambda_function=plan_fn,
                payload=sfn.TaskInput.from_object({"mode": mode, "run_id.$": "$.run_id",
                                                   "arch.$": "$.arch"}),
                result_selector={"run_id.$": "$.Payload.run_id", "arch.$": "$.Payload.arch",
                                 "action.$": "$.Payload.action", "size.$": "$.Payload.size",
                                 "spec.$": "$.Payload.spec"},
                retry_on_service_exceptions=True,
            )

        plan = invoke("Plan", "plan")
        finalize = invoke("Finalize", "finalize")
        halt = invoke("Halt", "halt")

        def submit(arch, array):
            # Batch array jobs need at least 2 children; single-probe waves
            # (the baseline, most bisection rounds) go as a plain job, and the
            # worker then runs the whole wave.
            return tasks.BatchSubmitJob(
                self, f"SubmitWave_{arch}_{'array' if array else 'single'}",
                job_name=f"bs-wave-{arch}",
                job_queue_arn=queues[arch].job_queue_arn,
                job_definition_arn=job_defs[arch].job_definition_arn,
                array_size=sfn.JsonPath.number_at("$.size") if array else None,
                payload=sfn.TaskInput.from_object({"spec": sfn.JsonPath.string_at("$.spec")}),
                integration_pattern=sfn.IntegrationPattern.RUN_JOB,
                result_path=sfn.JsonPath.DISCARD,
            )

        branches = {}
        for arch in ("arm64", "x86"):
            single, array = submit(arch, False), submit(arch, True)
            single.next(plan)
            array.next(plan)
            branches[arch] = (sfn.Choice(self, f"WaveSize_{arch}")
                              .when(sfn.Condition.number_greater_than("$.size", 1), array)
                              .otherwise(single))
        route = (sfn.Choice(self, "SubmitWave")
                 .when(sfn.Condition.string_equals("$.arch", "x86"), branches["x86"])
                 .otherwise(branches["arm64"]))
        decide = (sfn.Choice(self, "Decide")
                  .when(sfn.Condition.string_equals("$.action", "probe"), route)
                  .when(sfn.Condition.string_equals("$.action", "halted"), halt)
                  .otherwise(finalize))
        plan.next(decide)

        sm_log = logs.LogGroup(self, "RunLogs", log_group_name="/blindspot/runs",
                               retention=logs.RetentionDays.ONE_WEEK,
                               removal_policy=cdk.RemovalPolicy.DESTROY)
        self.state_machine = sfn.StateMachine(
            self, "Runs", state_machine_name="bs-runs",
            definition_body=sfn.DefinitionBody.from_chainable(plan),
            timeout=cdk.Duration.hours(1),
            logs=sfn.LogOptions(destination=sm_log, level=sfn.LogLevel.ERROR),
            tracing_enabled=True,
        )

        # ---- cost guard (rule C3): alarm on this project's tag only ----
        email = self.node.try_get_context("budget_email")
        if email:
            budgets.CfnBudget(
                self, "Budget",
                budget=budgets.CfnBudget.BudgetDataProperty(
                    budget_name="bs-monthly", budget_type="COST", time_unit="MONTHLY",
                    budget_limit=budgets.CfnBudget.SpendProperty(amount=BUDGET_USD, unit="USD"),
                    cost_filters={"TagKeyValue": ["user:project$blindspot"]},
                ),
                notifications_with_subscribers=[
                    budgets.CfnBudget.NotificationWithSubscribersProperty(
                        notification=budgets.CfnBudget.NotificationProperty(
                            comparison_operator="GREATER_THAN", notification_type=kind,
                            threshold=pct, threshold_type="PERCENTAGE"),
                        subscribers=[budgets.CfnBudget.SubscriberProperty(
                            subscription_type="EMAIL", address=email)],
                    ) for kind, pct in (("ACTUAL", 50), ("ACTUAL", 100), ("FORECASTED", 100))
                ],
            )

        cdk.CfnOutput(self, "StateMachineArn", value=self.state_machine.state_machine_arn)
        cdk.CfnOutput(self, "BucketName", value=bucket.bucket_name)
