import asyncio
import math
import logging
import numpy as np


from optibook.exchange_client import ORDER_TYPE_IOC  # noqa: E402
from optibook.exchange_client import InfoClient, ExecClient  # noqa: E402


logger = logging.getLogger(__name__)


TRADING_DAYS_PER_YEAR = 365
SECONDS_PER_DAY = 60 * 60 * 24

TARGET = 'TSLA_VOL'
MAX_POSITION = 100
SLEEP_SECS = 1


# TODO: tweak this
# Percentage edge that would make you want to go max_position
LIMIT = 0.05


def round_down_to_tick(price, tick_size):
    return math.floor(price / tick_size) * tick_size


def round_up_to_tick(price, tick_size):
    return math.ceil(price / tick_size) * tick_size


def _get_price(i, ins_id):
    book = i.get_last_price_book(ins_id)
    if book and book.bids and book.asks:
        top_bid, top_ask = book.bids[0].price, book.asks[0].price
        mid = (top_bid + top_ask) / 2.0
        return mid
    else:
        return None


def _get_desired_position(y, y_hat, max_position):
    y_hat = max(y_hat, 0)
    perc_edge = abs(y - y_hat) / y

    x = min(perc_edge / LIMIT, 1)
    x = np.sqrt(x)
    desired_position = round(x * max_position)

    side = 1 if y_hat > y else -1
    return desired_position * side


async def main(i, e, model, max_position, sleep_secs):
    target_tick = i.get_instruments()[TARGET].tick_size
    while True:
        missing_price = False
        prediction = model['intercept']
        for feature in model['features']:
            coef, instrument_id = feature[:2]
            transformation = feature[-1]

            if instrument_id == 'COMBINATION':
                instruments = feature[2]
                prices = []
                for instrument_id in instruments:
                    price = _get_price(i, instrument_id)
                    if price is None:
                        missing_price = True
                        logger.info(f'Missing price for: {instrument_id}. Not making prediction.')
                        break
                    prices.append(price)
                if not missing_price:
                    prediction += coef * transformation(*prices)
            else:
                price = _get_price(i, instrument_id)
                if price is None:
                    missing_price = True
                    logger.info(f'Missing price for: {instrument_id}. Not making prediction.')
                    break
                prediction += coef * transformation(price)

        if not missing_price:
            prediction = model.get('prediction_inverse_transform', lambda x: x)(prediction)
            logger.info(f'Final prediction: {prediction}')

            target = _get_price(i, TARGET)
            if target is not None:
                logger.info(f'Found target price: {target}')

                desired_position = _get_desired_position(target, prediction, max_position)
                position = e.get_positions()[TARGET]
                volume = desired_position - position
                logger.info(
                    f'Trading volume: {volume} to get from {position} to {desired_position}'
                )

                if volume != 0:
                    side = 'bid' if volume > 0 else 'ask'
                    if volume > 0:
                        if prediction > target:
                            price = round_down_to_tick(prediction, target_tick)
                        else:
                            price = round_down_to_tick(target * 1.1, target_tick)
                    else:
                        if prediction < target:
                            price = round_up_to_tick(prediction, target_tick)
                        else:
                            price = round_up_to_tick(target * 0.9, target_tick)

                    logger.info(f'Inserting a(n) {side} for {abs(volume)} lots @ {price}')
                    await e.insert_order(
                        instrument_id=TARGET,
                        price=price,
                        volume=abs(volume),
                        side=side,
                        order_type=ORDER_TYPE_IOC,
                    )
            else:
                logger.info(f'Missing target price ({TARGET}). Not trading.')

        logger.info(f'Sleeping for {sleep_secs} secs.')
        await asyncio.sleep(sleep_secs)


async def _trade_prediction(model):
    i = InfoClient()
    e = ExecClient()

    await i.connect()
    await e.connect()

    await main(
        i=i,
        e=e,
        model=model,
        max_position=MAX_POSITION,
        sleep_secs=SLEEP_SECS,
    )


def trade_prediction(model):
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(_trade_prediction(model))
    except KeyboardInterrupt:
        logger.info('Stopped (KeyboardInterrupt)')
