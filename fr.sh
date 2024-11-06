#!/bin/bash

# Function to load ignore patterns into arrays
load_ignore_patterns() {
    mapfile -t gitignore_patterns < .gitignore 2>/dev/null
    if [ -f ".flattenignore" ]; then
        mapfile -t flattenignore_patterns < .flattenignore
    fi
}

# Function to check if a file/folder should be ignored
should_ignore() {
    local item=$1
    local base_item=$(basename "$item")
    
    [ "$base_item" = "fr.sh" ] && return 0
    
    for pattern in "${gitignore_patterns[@]}" "${flattenignore_patterns[@]}"; do
        [[ -n "$pattern" && "$base_item" =~ ^${pattern}$ ]] && return 0
    done
    
    return 1
}

# Function to generate YAML representation of the file structure
generate_yaml() {
    local dir=$1
    local indent=$2
    local parent_path=$3
    
    # Use a single find command and process its output
    find "$dir" -mindepth 1 -maxdepth 1 ! -path '*/\.*' -printf '%y %p\n' | while read -r type path; do
        local base_item=$(basename "$path")
        local relative_path="$parent_path/$base_item"
        
        should_ignore "$path" && continue
        
        if [ "$type" = "d" ]; then
            echo "${indent}- path: $relative_path"
            echo "${indent}  type: directory"
            echo "${indent}  contents:"
            generate_yaml "$path" "  $indent" "$relative_path"
        else
            echo "${indent}- path: $relative_path"
            echo "${indent}  type: file"
        fi
    done
}

# Function to print file contents (only used with --ffc flag)
print_file_contents() {
    local file_path="${1#/}"
    [[ -d "$file_path" || ! -f "$file_path" || "$file_path" = "fr.sh" ]] && return
    
    if [[ "$file_path" =~ \.(py|js|ts|jsx|tsx|vue|rb|php|java|go|rs|c|cpp|h|hpp|cs|swift|kt|scala|html|css|scss|less|md|txt|sh|bash|zsh|json|yaml|yml|xml|sql|graphql|r|m|f|f90|jl|lua|pl|pm|t|ps1|bat|asm|s|nim|ex|exs|clj|lisp|hs|erl|elm)$ ]]; then
        {
            echo "<$file_path>"
            cat "$file_path"
            echo ""
            echo "</$file_path>"
            echo ""
        } >> "$flattened_file"
    fi
}

# Function to format time
format_time() {
    local milliseconds=$1
    local seconds=$(( milliseconds / 1000 ))
    local ms=$(( milliseconds % 1000 ))
    printf "%d.%03d seconds" "$seconds" "$ms"
}

# Main execution
cd "$(dirname "$0")"
output_file="repo_structure.yaml"
flattened_file="flattened_repo.txt"

# Load ignore patterns once at the start
declare -a gitignore_patterns flattenignore_patterns
load_ignore_patterns

# Start timing YAML generation
echo "Generating YAML structure..."
start_time=$(date +%s%N)

# Generate YAML structure
> "$output_file"
generate_yaml . "  " "" > "$output_file"

# Calculate YAML generation time
end_time=$(date +%s%N)
yaml_time=$(( (end_time - start_time) / 1000000 ))
echo "YAML file structure created in $(format_time $yaml_time)"
echo "Output saved as $output_file"

# Handle --ffc flag
if [[ "$1" == "--ffc" ]]; then
    echo "Flattening repository..."
    start_time=$(date +%s%N)
    
    > "$flattened_file"
    grep -E "^[[:space:]]*-[[:space:]]*path:" "$output_file" | 
    sed 's/^[[:space:]]*-[[:space:]]*path:[[:space:]]*//g' |
    while IFS= read -r file_path; do
        print_file_contents "$file_path"
    done
    
    # Calculate flattening time
    end_time=$(date +%s%N)
    flatten_time=$(( (end_time - start_time) / 1000000 ))
    echo "Repository flattened in $(format_time $flatten_time)"
    echo "Flattened content saved as $flattened_file"
    
    # Show total time
    total_time=$(( yaml_time + flatten_time ))
    echo "Total execution time: $(format_time $total_time)"
else
    echo "Repository structure created. Use --ffc flag to also flatten the file contents."
fi
