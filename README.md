# uda_aerial_semantic_segmentation_research

## AI-Assisted Development

This project leverages AI tools with specific configuration files:

- `.cursorrules` - Customizes Cursor Composer's code generation
- `repostructure.yaml` - Provides repository structure context

Additionally uses [Flatten](https://github.com/TrelisResearch/flatten) for maintaining codebase context:

```bash
curl -O https://raw.githubusercontent.com/TrelisResearch/flatten/main/fr.sh
chmod +x fr.sh
./fr.sh  # Generate structure
./fr.sh --ffc  # Generate and flatten
```

## Testing

Run all test suites:
```bash
python -m src.test_system
```

Run individual test suites:
```bash
python -m src.test_system data_loading
python -m src.test_system model_creation
python -m src.test_system loss_functions
python -m src.test_system logging
python -m src.test_system training
python -m src.test_system model_io
python -m src.test_system prediction
python -m src.test_system domain_adaptation
python -m src.test_system target_dataset
python -m src.test_system holyrood
python -m src.test_system adversarial_training
python -m src.test_system phase_management
python -m src.test_system fine_tuning
python -m src.test_system unsupervised_training
```

Run multiple specific test suites:
```bash
python -m src.test_system data_loading model_creation training
```

# Project Files

## File Listing

* .cursorrules - Configuration file for Cursor Composer behavior (Modified: 2024-03-21)
* instructions.txt - Log of all user prompts and interactions (Created: 2024-03-21)
* README.md - Project documentation and file listing (Created: 2024-03-21)
* summary.txt - Running summary of project evolution and key points (Created: 2024-03-21)
