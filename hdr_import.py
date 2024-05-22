"""
File: hdr_import.py

A parser for HDR ALS STXM software. The parser is based on the pyparsing library.

Author: Igor Beinik
Date: 2024-01-27
"""

import os, glob
from pyparsing import *
import ast
import numpy as np

class HDRFile:
    """HDR file import."""
    
    # hdr syntax features
    main_identifier = Word(alphanums+'_')
    identifier = Word(alphanums+'_')
    EQ = Literal('=')
    string_literal = quotedString.addParseAction(removeQuotes)
    lbrk = Literal('{')
    value = Word(alphas+'_')|string_literal|Word(nums+'.'+'-'+'e'+'+')
    experession = identifier + Suppress(EQ) + value + Suppress(';')
    points_entry = Group(main_identifier + Suppress(EQ) + Suppress('(') + delimitedList(value, delim=',').setParseAction(tokenMap(float)) + Suppress(')') + Suppress(';'))
    entry_content = OneOrMore(Dict(Group(experession)|points_entry)|value)
    rbrk = Literal('}')
    
    std_entry = Dict(Group(main_identifier + Suppress(EQ) +(lbrk + entry_content + rbrk + Suppress(';'))))
    short_entry = Dict(Group(main_identifier + Suppress(EQ) + value + Suppress(';')))
    single_region= Dict(Group(Word(nums) + Suppress(',') + Suppress('{') + OneOrMore(short_entry) + Suppress('}')))
    region_entry = Dict(Group(main_identifier + Suppress(EQ) + Suppress('(') + OneOrMore(single_region) + Suppress(')') + Suppress(';')))
    enrg_entry = Group(lbrk + entry_content + rbrk)
    enrg_regions_entry = Dict(Group(main_identifier + Suppress(EQ) + Suppress('(') + delimitedList(Word(nums)|enrg_entry, delim=',') + Suppress(')') + Suppress(';')))
    img_entry = Dict(Group(main_identifier + Suppress(EQ) + lbrk + OneOrMore(Dict(Group(experession))|enrg_regions_entry) + rbrk + Suppress(';')))
    comp_entry = Dict(Group(main_identifier + Suppress(EQ) + Suppress('{') + OneOrMore(std_entry|short_entry|region_entry) + Suppress('}')))
    ext_region = Dict(Group(main_identifier + Suppress(EQ) + Suppress('(') + OneOrMore(Dict(Group(Word(nums) + Suppress(',') + Suppress('{') + OneOrMore(std_entry) + Suppress('}'))))+ Suppress(')')+ Suppress(';')))
    definition_entry = Dict(Group(main_identifier + Suppress(EQ) +(lbrk + OneOrMore(short_entry|ext_region|std_entry|region_entry|comp_entry) + rbrk + Suppress(';'))))
    date_entry = Dict(Group(Word(alphas) + Suppress(EQ) + QuotedString('"') + Suppress(';')))
    bool_entry = Dict(Group(Word(alphanums+'_') + Suppress(EQ) + oneOf("true false") + Suppress(';')))
    
    entry = OneOrMore(definition_entry|comp_entry|date_entry|bool_entry|region_entry|short_entry|img_entry)
    
    def __init__(self, file_path):
            self.base_name, self.ext = os.path.splitext(file_path)
            # Check if a corresponding .hdr file exists and read it
            if os.path.isfile(self.base_name + '.hdr'):
                print('Reading hdr file:', self.base_name + '.hdr')
                with open(self.base_name + '.hdr') as file:
                    self.hdr_content_raw = file.read()
                    # Assuming self.entry.parseString is defined elsewhere in your class
                    self.parsing_result = self.entry.parseString(self.hdr_content_raw)
                    self.as_dict = self.parsing_result.asDict()
                    # Assuming _eval_all_values is a method defined in your class
                    self._eval_all_values(self.as_dict)
                # Load a single .xim file if the class is initialized with a specific file path
                with open(self.base_name + '_a.xim') as xim_file:
                    print('Reading xim file:', self.base_name + '_a.xim')
                    self.xim = np.loadtxt(self.base_name + '_a.xim', dtype=int)
            # If a directory is provided, process all matching .xim files within it
            elif os.path.isdir(self.base_name):
                self.base_file_name = self.base_name.split('/')[-1]
                self.xim = []
                self.sort_tool = []
                print('Reading hdr file:', self.base_name + '/' + self.base_file_name + '.hdr')
                with open(self.base_name + '/' + self.base_file_name + '.hdr') as file:
                    self.hdr_content_raw = file.read()
                    self.parsing_result = self.entry.parseString(self.hdr_content_raw)
                    self.as_dict = self.parsing_result.asDict()
                    self._eval_all_values(self.as_dict)
                print('Reading xim file(s):')
                # Collect filenames and their corresponding numbers
                filenames_and_numbers = []
                for filename in glob.iglob(f'{self.base_name}/*_a*'):
                    base, ext = os.path.splitext(filename)
                    xim_number = int(base.split('_')[-1][1:])
                    filenames_and_numbers.append((xim_number, filename))
                # Sort by the extracted numbers
                filenames_and_numbers.sort(key=lambda x: x[0])
                # Load the files in sorted order
                for xim_number, filename in filenames_and_numbers:
                    print(filename)  # This prints the sorted filenames
                    self.sort_tool.append(xim_number)
                    self.xim.append(np.loadtxt(filename, dtype=int))

                #self.base_name, self.ext = os.path.splitext(file_path)
                #print('The base filename is:', self.base_name)
            
    def _tryeval(self, val):
        """Evaluates the type."""
        try:
            val = ast.literal_eval(val)
        except ValueError as err:
            if val == 'true':
                return True
            elif val == 'false':
                return False
            elif isinstance(val, list):
                print(val)
            else:
                return val
        except SyntaxError as synt_err:
            return str(val)
        return val
    
    def _eval_all_values(self, nested_dictionary):
        """Traversing the values of the dict to get the right types."""
        for key, value in nested_dictionary.items():
            if type(value) is dict:
                self._eval_all_values(value)
            else:
                nested_dictionary[key] = self._tryeval(str(value))
