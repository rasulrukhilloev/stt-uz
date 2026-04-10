from __future__ import annotations

import logging

from telegram.ext import ApplicationBuilder

from app.bot.handlers import register_handlers
from app.config import Settings
from app.services import AppServices


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def main() -> None:
    configure_logging()
    settings = Settings.from_env()
    services = AppServices.build(settings)

    application = ApplicationBuilder().token(settings.telegram_bot_token).build()
    application.bot_data["services"] = services
    register_handlers(application)
    application.run_polling()


if __name__ == "__main__":
    main()

