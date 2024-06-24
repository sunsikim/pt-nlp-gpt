import typer
import jobs
from jobs.configure import ExecutableJobs
from typing import Annotated
from typer import Typer

app = Typer()


@app.command("list-jobs")
def list_jobs():
    for executable_job in ExecutableJobs.__subclasses__():
        job_executor = executable_job()
        print(job_executor.name, job_executor.__doc__)


@app.command("run-job")
def run_job(
    job_prefix: Annotated[
        str,
        typer.Argument(
            envvar="APP_JOBS",
            show_envvar=True,
            show_default=True,
            help="string of job name to be executed",
        ),
    ] = "preprocess",
    data_type: Annotated[
        str,
        typer.Argument(
            envvar="DATA_TYPE",
            show_envvar=True,
            show_default=True,
            help="one of 'shakespeare', 'openwebtext'",
        ),
    ] = "shakespeare",
):
    # register executable jobs
    executable_jobs = {}
    for job in ExecutableJobs.__subclasses__():
        job_executor = job()
        if job_executor.name in executable_jobs:
            raise ValueError(
                f"Found multiple jobs with duplicated name '{job_executor.name}'"
            )
        else:
            executable_jobs[job_executor.name] = job_executor.execute

    # execute input jobs
    job_name = f"{job_prefix}-{data_type}"
    if job_name not in executable_jobs:
        raise ValueError(f"Job with name '{job_name}' is not registered")
    else:
        executable_jobs[job_name]()
