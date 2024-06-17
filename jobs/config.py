import logging

logging.basicConfig(
    format="%(asctime)s (%(module)s) (%(funcName)s) : %(msg)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


class ExecutableJobs:

    def __init__(self, job_name: str = None):
        self.name = job_name

    def execute(self):
        raise NotImplementedError
