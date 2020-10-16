DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
conda env export | grep -v "^prefix: " > "$DIR/"environment.yml