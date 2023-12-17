SCRIPT_DIR="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

for inputfile in $SCRIPT_DIR/../assets/*.BMP ; do
    outputfile="${inputfile%.*}.png"
    convert "$inputfile" "$outputfile"
done
