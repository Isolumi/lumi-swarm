#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

mkdir -p "${tmp_dir}/bin"

cat >"${tmp_dir}/bin/git" <<'GIT'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "${GIT_LOG}"
GIT
chmod +x "${tmp_dir}/bin/git"

export GIT_LOG="${tmp_dir}/git.log"
export PATH="${tmp_dir}/bin:${PATH}"

"${repo_root}/scripts/save.sh" add save script

expected="${tmp_dir}/expected.log"
cat >"${expected}" <<'EXPECTED'
add .
commit -m add save script
push
EXPECTED

diff -u "${expected}" "${GIT_LOG}"

if "${repo_root}/scripts/save.sh" >"${tmp_dir}/no-message.out" 2>&1; then
    echo "save.sh should fail without a commit message"
    exit 1
fi

grep -q "Usage:" "${tmp_dir}/no-message.out"
