REPOSITORIES=$1
COMMITS=$2
SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")

echo "Repositories are " ${REPOSITORIES}
echo "Commits are " ${COMMITS}
echo "Script directory is " ${SCRIPT_DIR}

python3 ${SCRIPT_DIR}/download.py ${REPOSITORIES}
python3 ${SCRIPT_DIR}/extract.py ${REPOSITORIES} ${COMMITS}
python3 ${SCRIPT_DIR}/count_method_pairs.py
python3 ${SCRIPT_DIR}/beautify_js.py 
python3 ${SCRIPT_DIR}/extract_segment_pairs.py 
python3 ${SCRIPT_DIR}/count_different_segments.py
python3 ${SCRIPT_DIR}/preprocess.py 
python3 ${SCRIPT_DIR}/split_data.py 
cat testing/* > test.csv
cat training/* > train.csv
cat validation/* > valid.csv
python3 ${SCRIPT_DIR}/check_for_overlaps.py 
python3 ${SCRIPT_DIR}/create_java_only.py 
