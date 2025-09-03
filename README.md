# Gene Sequence Analysis

This repository contains tools for analyzing and comparing gene sequences from gel electrophoresis images.

## Example Input

Input is a gel electrophoresis image (JPG/PNG) showing DNA bands, for example:

```
[Gel Image]
1  2  3  4  5  6  7  8  9 10 11
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
```

## Example Output

1. **Visual Output**:
   - Original gel image with detected lanes
   - Similarity matrix heatmap

2. **Textual Output**:
   ```
   Analysis complete! Results saved to gel_analysis.png

   Similarity Matrix (lower values = more similar):
   [[0.   0.26 0.24 0.11 0.11 0.17 0.24 0.24 0.27 0.29 0.97]
    [0.26 0.   0.07 0.09 0.12 0.1  0.11 0.11 0.11 0.12 1.01]
    ...
    [0.97 1.01 1.   1.02 0.99 1.05 1.07 1.08 1.14 1.1  0.  ]]
   ```

## Features

- **Gel Image Analysis**: Process gel electrophoresis images to detect and compare DNA bands
- **Sequence Comparison**: Compare DNA sequences and calculate similarity scores
- **Visualization**: Generate visual representations of sequence comparisons

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd gene_sequence
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Gel Image Analysis

To analyze a gel electrophoresis image:
```bash
python gel_analyzer.py path_to_your_image.jpg
```

### Sequence Comparison

To compare DNA sequences:
```bash
python sequence_comparison.py
```

## Project Structure

- `gel_analyzer.py`: Analyzes gel electrophoresis images and generates similarity reports
- `sequence_comparison.py`: Compares DNA sequences and calculates alignment scores
- `requirements.txt`: Lists required Python packages
- `.gitignore`: Specifies files to ignore in version control
- `gel_analysis.png`: Example output image (generated after analysis)
- `sequence_comparison.png`: Example sequence alignment visualization

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
