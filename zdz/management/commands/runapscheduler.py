import logging

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from django.conf import settings
from django.core.management.base import BaseCommand
from django_apscheduler import util
from django_apscheduler.jobstores import DjangoJobStore
from django_apscheduler.models import DjangoJobExecution

from zdz.schedulers.external_data_sync import sync_station_data
from zdz.schedulers.external_ec_upload import ec_upload

logger = logging.getLogger(__name__)


@util.close_old_connections
def delete_old_job_executions(max_age=604_800):
    DjangoJobExecution.objects.delete_old_job_executions(max_age)


class Command(BaseCommand):
    help = "Runs APScheduler."

    def handle(self, *args, **options):
        scheduler = BlockingScheduler(timezone=settings.TIME_ZONE)
        scheduler.add_jobstore(DjangoJobStore(), "default")

        scheduler.add_job(
            sync_station_data,
            trigger=CronTrigger(
                hour="*/1", minute="5"
            ),
            id="sync_station_data",  # The `id` assigned to each job MUST be unique
            max_instances=1,
            replace_existing=True,
            misfire_grace_time=3600
        )
        scheduler.add_job(
            ec_upload,
            trigger=CronTrigger(
                hour="*/1", minute="14"
            ),
            id="ec_upload",  # The `id` assigned to each job MUST be unique
            max_instances=1,
            replace_existing=True,
            misfire_grace_time=3600
        )
        logger.info("Added job 'test_job'.")

        scheduler.add_job(
            delete_old_job_executions,
            trigger=CronTrigger(
                day_of_week="mon", hour="00", minute="00"
            ),  # Midnight on Monday, before start of the next work week.
            id="delete_old_job_executions",
            max_instances=1,
            replace_existing=True,
        )
        logger.info(
            "Added weekly job: 'delete_old_job_executions'."
        )

        try:
            logger.info("Starting scheduler...")
            scheduler.start()
        except KeyboardInterrupt:
            logger.info("Stopping scheduler...")
            scheduler.shutdown()
            logger.info("Scheduler shut down successfully!")
