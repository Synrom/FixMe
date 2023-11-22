BUGIDS_DIR=$1
BUGGY_REPOS_DIR=$2

echo "Bug IDs directory is " ${BUGIDS_DIR}
echo "Buggy Repositories directory is " ${BUGGY_REPOS_DIR}

mkdir ${BUGIDS_DIR}
mkdir ${BUGGY_REPOS_DIR}

for pid in $(defects4j pids); do
    bids_output=$(defects4j query -p "$pid" -q "classes.modified")
    echo "$bids_output" > "${BUGIDS_DIR}/${pid}.classes"


    mkdir -p "${BUGGY_REPOS_DIR}/$pid/$id/"

    for id in $(defects4j bids -p "$pid" -A); do
        defects4j checkout -p "$pid" -v ${id}b -w "${BUGGY_REPOS_DIR}/$pid/$id/buggy"
        defects4j checkout -p "$pid" -v ${id}f -w "${BUGGY_REPOS_DIR}/$pid/$id/fixed"
        defects4j checkout -p "$pid" -v ${id}b -w "${BUGGY_REPOS_DIR}/$pid/$id/bugs2fix"

        srcdir=$(defects4j export -p dir.src.classes -w "${BUGGY_REPOS_DIR}/$pid/$id/buggy")
        echo "$srcdir" > "${BUGIDS_DIR}/${pid}_${id}.buggy"

        srcdir=$(defects4j export -p dir.src.classes -w "${BUGGY_REPOS_DIR}/$pid/$id/fixed")
        echo "$srcdir" > "${BUGIDS_DIR}/${pid}_${id}.fixed"

    done
done
