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


@app.command("run-jobs")
def run_jobs(
    jobs: Annotated[
        str,
        typer.Argument(
            envvar="APP_JOBS",
            show_envvar=True,
            show_default=True,
            help="comma-separated text of job names to be executed",
        ),
    ] = "preprocess,train,evaluate",
):
    # register executable jobs
    executable_jobs = {}
    for job in ExecutableJobs.__subclasses__():
        _job = job()
        if _job.name in executable_jobs:
            raise ValueError(f"Found multiple jobs with duplicated name '{_job.name}'")
        else:
            executable_jobs[_job.name] = _job.execute

    # execute input jobs
    for job_name in jobs.split(","):
        if job_name not in executable_jobs:
            raise ValueError(f"Job with name '{job_name}' is not registered")
        else:
            executable_jobs[job_name]()
