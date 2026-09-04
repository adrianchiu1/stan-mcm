"""Validation machinery in the package (S6 WP3): the generic SBC engine
(:mod:`.sbc`), the generic parameter-recovery arithmetic (:mod:`.recovery`),
and the gate-suite runner + report (:mod:`.suite`) behind ``mtk validate
<family>`` / ``api.validate``. Families declare their gate designs in
``macrotoolkit/families/<family>_validation.py`` and register them via
``FamilyEntry.validation_suite``."""
