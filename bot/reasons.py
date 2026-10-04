"""The reasons attached to each target weight, recorded with every decision and order."""

ENTRY = "entry"
HOLD = "hold"
CORE = "core"
EXIT_TREND = "exit_trend"
EXIT_STOP = "exit_stop"
EXIT_REGIME = "exit_regime"
HOLD_NO_DATA = "hold_no_data"
HOLD_HALTED = "hold_halted"   # Roostoo is not trading the pair: hold it as it is
SHORT_ENTRY = "short_entry"
SHORT_HOLD = "short_hold"
EXIT_SHORT_STOP = "exit_short_stop"
EXIT_SHORT = "exit_short"
ROTATION = "rotation"
LS_LONG = "ls_long"           # the long-short trend book (research H50): coin in an uptrend
LS_SHORT = "ls_short"         # ... and in a downtrend
