#
# Copyright (C) 2013 - 2021 Oracle and/or its affiliates. All rights reserved.
#


class PgxRedactionRuleConfig:
    """A class representing redaction rule configurations."""

    _java_class = 'oracle.pgx.config.PgxRedactionRuleConfig'

    def __init__(self, java_redaction_rule_config) -> None:
        self._redaction_rule_config = java_redaction_rule_config

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._redaction_rule_config.equals(other._redaction_rule_config)
