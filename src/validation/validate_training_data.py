"""
Validates the raw training table before it's trusted for anything
downstream. Run manually or as a CI step: python -m src.validation.validate_training_data
"""

import logging
import sys

import great_expectations as gx
import pandas as pd

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)


def validate_ml_table(path=None) -> bool:
    path = path or PROJECT_ROOT / "artifacts" / "ml_table_orders.parquet"
    df = pd.read_parquet(path)

    context = gx.get_context(mode="ephemeral")

    data_source = context.data_sources.add_pandas("pandas_source")
    data_asset = data_source.add_dataframe_asset(name="ml_table")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("ml_table_batch")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    suite = context.suites.add(gx.ExpectationSuite(name="ml_table_suite"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="order_id"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeUnique(column="order_id"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="customer_state"))
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToMatchRegex(column="customer_state", regex=r"^[A-Z]{2}$")
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="total_price", min_value=0, strict_min=True
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(column="total_freight_value", min_value=0)
    )

    result = batch.validate(suite)

    if not result.success:
        logger.error("data validation FAILED: %s", result)
        return False

    logger.info("data validation passed: %d expectations checked", len(result.results))
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ok = validate_ml_table()
    sys.exit(0 if ok else 1)
