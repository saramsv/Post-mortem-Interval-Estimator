cat PMIs | grep [0-9][a-z]006[0-9].*:0 | cut -d '/' -f10 | sort -u > june_cases
bash grep.sh june_cases PMIs > june_cases_PMI
cat june_cases_PMI | shuf > june_cases_PMI_shuf
