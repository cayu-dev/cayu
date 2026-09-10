"""Fixed guest-side probes shared by POSIX execution backends."""

# Resolve an external executable without invoking it. `command -v` also
# accepts shell functions/builtins, which process-form guest dispatch cannot
# execute. Preserve empty/relative PATH entries and do not split on whitespace.
EXECUTABLE_AVAILABILITY_SCRIPT = r"""
name=$1
case "$name" in
    */*) [ -f "$name" ] && [ -x "$name" ]; exit $? ;;
esac
remaining=${PATH-/bin:/usr/bin}
while :; do
    directory=${remaining%%:*}
    [ -n "$directory" ] || directory=.
    if [ -f "$directory/$name" ] && [ -x "$directory/$name" ]; then
        exit 0
    fi
    case "$remaining" in
        *:*) remaining=${remaining#*:} ;;
        *) exit 1 ;;
    esac
done
"""
