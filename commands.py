import typer
import jobs
from jobs.config import ExecutableJobs
from typing import Annotated
from typer import Typer

app = Typer()


@app.command("list-jobs")
def list_jobs():
    for executable_job in ExecutableJobs.__subclasses__():
        job_executor = executable_job()
        print(job_executor.name, job_executor.__doc__)


@app.command("run-jobs")
def run_jobs(
    job_names: Annotated[
        str,
        typer.Option(
            envvar="APP_JOBS",
            show_envvar=True,
            show_default=True,
            help="comma-separated text of job names to be executed",
        ),
    ] = "preprocess,train"
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
    for job_name in job_names.split(","):
        if job_name not in executable_jobs:
            raise ValueError(f"Job with name '{job_name}' is not registered")
        else:
            executable_jobs[job_name]()
