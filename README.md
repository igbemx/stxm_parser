# STXM HDR File Parser

This repository contains a file parser that defines a class named `HDRFile` for parsing HDR files used by the ALS STXM software. The parser utilizes the `pyparsing` library.

## Functionality

The `HDRFile` class provides the following functionality:

- Parses an HDR file based on its specific syntax rules.
- Identifies different entry types within the file, including:
    - Definitions (key-value pairs with curly braces)
    - Components (collections of definitions)
    - Dates
    - Boolean values
    - Regions
    - Short entries (key-value pairs with semicolon)
    - Images
    - Points entries (key-value pairs with points list)
    - Energy entries
    - Energy region entries
- Converts the parsed data into a nested dictionary structure.
- Optionally evaluates string values that might represent numbers or booleans.

## Requirements

- Python 3.x
- `pyparsing` library
- `ast` library (for `ast.literal_eval`)
- `numpy` library (for loading XIM files, might be optional)

## Usage

To use the `HDRFile` class:

1. Import the `HDRFile` class from `hdr_import.py`.
2. Create an instance of `HDRFile` by providing the path to the HDR file (or directory containing HDR and corresponding XIM files).
     - If a directory path is provided, the class will process all matching .xim files within it.
3. Access the parsed data through the `as_dict` attribute (a dictionary).

