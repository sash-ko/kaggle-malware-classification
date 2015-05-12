import numpy as np
from collections import Counter
import re

g_hex_values = {hex(i)[2:].zfill(2) for i in range(256)}

NUM_FEATURES = 7

def detect_code_sections(lines):
    code = []
    code_sections = set()
    for l in lines:
        # most frequent
        if l.startswith('.text:'):
            code.append(l)
        else:
            found = False
            for cs in code_sections:
                if l.startswith(cs):
                    code.append(l)
                    found = True
                    break
            if not found:
                if 'segment type: pure code' in l.replace('\t', ' '):
                    section = parse_section(l)
                    if section:
                        print 'found new code section: "%s"' % section
                        code_sections.add(section)
    return code

def find_imported_dlls(lines):
    dlls = set()
    for line in lines:
        if line.startswith('.text:'):
            continue
        line = line.replace('\t', ' ')
        if 'imports from' in line:
            dlls.add(line.split()[-1])
    return {d: 1 for d in dlls}
    #return sorted(list(dlls))

def parse(lines):
    lines = [l.strip().lower() for l in lines]
    sections = count_sections(lines)
    # dlls = find_imported_dlls(lines)
    code = detect_code_sections(lines)

    asm_misc = asm_misc_features(lines)

    if not code:
        print 'No code section found!'

    opcodes = parse_code(code)

    ngrams_1 = build_ngrams(opcodes, 1)
    ngrams_2 = build_ngrams(opcodes, 2)
    ngrams_3 = build_ngrams(opcodes, 3)
    ngrams_4 = build_ngrams(opcodes, 4)
    ngrams_5 = build_ngrams(opcodes, 5)

    feature_sets = [ngrams_1, ngrams_2, ngrams_3, ngrams_4, ngrams_5,
                    np.array(sections), np.array(asm_misc)
                    # np.array(dlls),
                    ]
    assert len(feature_sets) == NUM_FEATURES
    return feature_sets

def asm_misc_features(lines):
    keys = [" dd ",".data:",".dll",".exe",".idata:",".rdata:",".rsrc:",
            ".text:",".unisec","__cdecl","__clrcall","__ctype","__cxx",
            "__dllonexit","__fastcall","__imp_","__msg_","__stdcall",
            "__thiscall","add","address","align","alloc","arg_","astatus_",
            "attributes","bool","bp-based","byte","call","call    ds:",
            "callback","calloc","certfreecrlcontext","close","cmp","code",
            "code xref","collapsed","const","create","createtraceinstanceid",
            "critical","data xref","dec","descriptor","desired","destroy",
            "dll","dllentrypoint","ds:","ds:getpriorityclass","dwflags",
            "dword","dword_","dwprovtype","eax","ebp","ebx","ecx",
            "ehcookiexoroffset","endp","endtime","environment","error","esi",
            "esp","exception","extrn","failed","ffreep","file","finally",
            "flush","fmode","font","format","frame","free","fstp",
            "function chunk","gdi","global","gscookieoffset","handler","heap",
            "henhmetafile","hheap","hkey","hmodule","hwnd","icm","icode:",
            "idiv","import","imul","inc","init","insd","instancename","jle",
            "jmp","jnz","jumptable","jz","kernel","large","lea","load","loc_",
            "lpmem","lpvoid","lstrcata","malloc","memcpy","memory","meta",
            "microsoft","module","move","movsx","movzx","mutex","near","off_",
            "offset","operator new","outsd","pop","press ","private","proc",
            "properties","protected","ptr","public","push","push    ds:",
            "querytracew","qword","realloc","reg","rep","resource","retn",
            "rva","s u b r o u t i n e","sampletecriterface","scoperecord",
            "secur32.dll","security","short","size_t","sleep","software",
            "sp-analysis","src","starttime","status","std","std:","stosd",
            "strlen","struct","sub","sub_","switch","sysexit","system",
            "system32","szcontainer","szprovider","test","thread","throw",
            "tls","trace","user","var_","vftable","virtual","vlc_plugin_set",
            "void *","windows","winmain","xml"]

    featues = Counter()
    for line in lines:
        for key in keys:
            if key in line:
                featues[key] = featues[key] + 1
    return featues.items()

def parse_code(lines):
    opcodes = []
    current_block = []
    for line in lines:
        idx = line.find(' ')
        if idx != -1:
            line = line[idx + 1: ].strip()
            if line[0] != ';':
                cmd = parse_asm_command(line)
                if cmd:
                    current_block.append(cmd)
        else:
            if current_block:
                opcodes.append(current_block)

            current_block = []

    if current_block:
        opcodes.append(current_block)

    return opcodes

def build_ngrams(opcodes, n):
    counter = Counter()
    for group in opcodes:
        group = [g for g in group if g is not None]
        ngrams = make_ngrams(group, n)

        for ngram in ngrams:
            if ngram:
                counter[ngram] = counter[ngram] + 1

    return np.array(counter.items())

def make_ngrams(lst, n):
    ngrams = zip(*[lst[i:] for i in range(n)])
    if n == 1:
        ngrams = [n[0] for n in ngrams]

    return ngrams

def skip_hex_groups(groups):
    hex_groups = []
    for i in range(len(groups)):
        if not is_hex_group(groups[i]):
            break
        else:
            hex_groups.append(groups[i])
    return (' '.join(hex_groups), ' '.join(groups[i:]))

def ommit_comments(line):
    idx = line.find(';')
    if idx != -1:
        line = line[:idx]
    return line

def check_set_cmds(cmds, items):
    for cmd in cmds:
        if cmd in items:
            return cmd

def remove_other_hex(items):
    if items:
        for idx, val in enumerate(items):
            if val == '00':
                continue

            if val not in g_hex_values:
                break
        return items[idx: ]
    return []

def contains_numbers(cmd):
    return re.match('.*\d.*', cmd) is not None

def contains_less_sign(items):
    return len([i for i in items if i.startswith('<')]) > 0

def get_less_sign_cmd(items):
    indices = [idx for idx, c in enumerate(items) if c.startswith('<')]
    if indices:
        return items[indices[-1] - 1]

def split_by_underscore(val):
    if '_' in val:
        items = val.split('_', 1)
        if not contains_numbers(items[0]) and contains_numbers(items[1]):
            return items[0]

def strip_numbers(val):
    if val not in {'ud2', 'fldln2', 'fldlg2'}:
        val = val.rstrip('0123456789')
    return val

def parse_asm_command(line):
    line = ommit_comments(line)

    hex_groups, line = skip_hex_groups(line.split('\t'))
    if not hex_groups:
        return None

    items = line.replace('+', ' ').split()
    cmd = check_set_cmds(['dd', 'db', 'dw', 'dt', 'dq', 'do', 'ddq'], items)
    if cmd is None and items:
        items = remove_other_hex(items)
        hex_groups = hex_groups.split()
        cmd = items[0]
        if cmd in hex_groups or cmd == '=':
            cmd = None
        elif cmd and contains_numbers(cmd):
            new_cmd = split_by_underscore(cmd)
            if new_cmd is None:
                cmd = strip_numbers(cmd)
            else:
                cmd = new_cmd

        if cmd == '' or cmd == ';' or cmd in g_hex_values or\
                (cmd and cmd.startswith('.text:')) or\
                (cmd and (len(cmd) == 1 or cmd.endswith('h,'))):
            cmd = None

    return cmd

def is_hex_group(item):
    items = [i.rstrip('+') for i in item.split()]
    return all([i in g_hex_values for i in items])

def parse_section(line):
    idx = line.find(':')
    if idx != -1:
        return line[: idx + 1].lower()

def count_sections(lines):
    counter = Counter()
    for line in lines:
        section = parse_section(line)
        if section:
            counter[section] = counter[section] + 1
    return counter.items()
