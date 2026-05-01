"""
Exposes the main rule retrieval function `get_rules` used by the `Lakeflow`
pipelines.

This encapsulates the logic of aggregating rules from different business
domains, keeping the main pipeline script clean and focused purely on data
flow.
"""

###############################################################################
# Imports
###############################################################################

# Importamos reglas de seguros
from .policies import get_policy_rules
from .claims import get_claim_rules
from .labels import get_label_rules


###############################################################################
# Functions
###############################################################################

def _get_all_rules_as_list_of_dict():
    """
    Returns the complete catalog of data quality rules for the entire project,
    combining policies, claims, and labels into a single list.
    """
    all_rules = []
    all_rules.extend(get_policy_rules())
    all_rules.extend(get_claim_rules())
    all_rules.extend(get_label_rules())
    
    return all_rules


def get_rules(tag):
    """
    Loads data quality rules from the central repository matching the
    given tag.

    Returns a dictionary formatted specifically for the `@dp.expect_all`
    decorator, where the key is the rule name and the value is the `SQL`
    constraint.
    """
    return {
        row["name"]: row["constraint"]
        for row in _get_all_rules_as_list_of_dict()
        if row["tag"] == tag
    }