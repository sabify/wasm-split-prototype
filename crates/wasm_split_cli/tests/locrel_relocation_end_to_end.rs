//! Builds a minimal splittable wasm whose data holds a pointer that is
//! relative to its own location (`R_WASM_MEMORY_ADDR_LOCREL_I32`, the value
//! is `symbol address - address of the pointer`), next to an absolute one,
//! and runs the full `wasm_split_cli_support::transform()` pipeline on it.
//!
//! The pointers sit in a split module's symbol and refer to a symbol in the
//! main module, so the relocation moves both the pointer and its target.
//! Asserts that the absolute pointer holds the target's new address and the
//! relative one the difference to its own new address.
//!
//! `location_relative_pointer_to_an_unmoved_target_is_still_updated`: the
//! target is a zero-sized symbol, which is never relocated, but the pointer
//! moves. Asserts that the pointer is re-encoded against its new address.
//!
//! `location_relative_pointer_keeps_the_low_32_bits`: a large addend makes
//! the difference exceed the signed 32-bit range. Asserts that, like
//! `wasm-ld`, the low 32 bits are encoded rather than rejected.
//!
//! The input builder is shared with `shared_data_symbols_end_to_end.rs`.

use std::borrow::Cow;

use wasm_encoder::{
    CodeSection, ConstExpr, CustomSection, DataSection, Encode, ExportKind, ExportSection,
    Function, FunctionSection, ImportSection, Instruction, MemorySection, MemoryType, Module,
    RefType, TableSection, TableType, TypeSection, ValType,
};
use wasmparser::{DataKind, Operator, Parser, Payload};

use wasm_split_cli_support::{transform, Options};

#[path = "../src/magic_constants.rs"]
mod magic_constants;

const SPLIT_HASH: &str = "00000000000000000000000000000000";

/// Where the input's only data segment is placed in memory.
const SEGMENT_BASE: u32 = 1024;

/// `R_WASM_MEMORY_ADDR_LEB`
const R_WASM_MEMORY_ADDR_LEB: u8 = 3;
/// `R_WASM_MEMORY_ADDR_I32`
const R_WASM_MEMORY_ADDR_I32: u8 = 5;
/// `R_WASM_MEMORY_ADDR_LOCREL_I32`: the symbol's address relative to the relocated bytes
const R_WASM_MEMORY_ADDR_LOCREL_I32: u8 = 23;

/// A defined data symbol: `(name, offset in the segment, size)`.
struct DataSymbol(&'static str, usize, usize);

/// Which module a function belongs to.
enum Owner {
    /// Exported from the main module under this name.
    Main(&'static str),
    /// The entry of the split with this name.
    Split(&'static str),
}

/// A defined function that refers to data symbols (by index into
/// [`Input::symbols`]) through `i32.const` immediates, in this order.
struct Func(Owner, Vec<usize>);

struct Input {
    data: Vec<u8>,
    /// log2 of the segment's alignment
    alignment: u32,
    symbols: Vec<DataSymbol>,
    funcs: Vec<Func>,
    /// Pointers inside the data: `(offset in the segment, index of the symbol pointed to,
    /// relocation type, addend)`. The builder writes the input value there.
    data_relocs: Vec<(usize, usize, u8, i32)>,
}

fn uleb(mut value: u32, out: &mut Vec<u8>) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

fn sleb(mut value: i32, out: &mut Vec<u8>) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        let done = (value == 0 && byte & 0x40 == 0) || (value == -1 && byte & 0x40 != 0);
        if done {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

fn name(s: &str, out: &mut Vec<u8>) {
    uleb(s.len() as u32, out);
    out.extend_from_slice(s.as_bytes());
}

/// `i32.const` with a 5-byte padded immediate, as emitted for relocatable code.
fn padded_i32_const(mut value: u32) -> [u8; 6] {
    let mut bytes = [0x41, 0, 0, 0, 0, 0];
    for byte in &mut bytes[1..5] {
        *byte = (value & 0x7f) as u8 | 0x80;
        value >>= 7;
    }
    bytes[5] = (value & 0x7f) as u8;
    bytes
}

fn split_import_name(split: &str) -> String {
    format!("__wasm_split_00{split}00_import_{SPLIT_HASH}")
}

fn split_export_name(split: &str) -> String {
    format!("__wasm_split_00{split}00_export_{SPLIT_HASH}")
}

impl Input {
    fn splits(&self) -> Vec<&'static str> {
        self.funcs
            .iter()
            .filter_map(|Func(owner, _)| match owner {
                Owner::Split(split) => Some(*split),
                Owner::Main(_) => None,
            })
            .collect()
    }

    /// Everything but the linking metadata. The metadata is appended
    /// afterwards, which keeps the code offsets computed from this prefix valid.
    fn build_module_prefix(&self) -> Vec<u8> {
        let splits = self.splits();
        let mut module = Module::new();

        let mut types = TypeSection::new();
        types.ty().function([], [ValType::I32]);
        module.section(&types);

        // One placeholder import per split, then the defined functions.
        let mut imports = ImportSection::new();
        for split in &splits {
            imports.import(
                magic_constants::PLACEHOLDER_IMPORT_MODULE,
                &split_import_name(split),
                wasm_encoder::EntityType::Function(0),
            );
        }
        module.section(&imports);

        let mut functions = FunctionSection::new();
        for _ in &self.funcs {
            functions.function(0);
        }
        module.section(&functions);

        let mut tables = TableSection::new();
        tables.table(TableType {
            element_type: RefType::FUNCREF,
            minimum: 1,
            maximum: Some(1),
            table64: false,
            shared: false,
        });
        module.section(&tables);

        let mut memories = MemorySection::new();
        memories.memory(MemoryType {
            minimum: 1,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        });
        module.section(&memories);

        let mut exports = ExportSection::new();
        for (i, Func(owner, _)) in self.funcs.iter().enumerate() {
            let func_index = (splits.len() + i) as u32;
            let export_name = match owner {
                Owner::Main(export) => export.to_string(),
                Owner::Split(split) => split_export_name(split),
            };
            exports.export(&export_name, ExportKind::Func, func_index);
        }
        exports.export("__indirect_function_table", ExportKind::Table, 0);
        exports.export("memory", ExportKind::Memory, 0);
        module.section(&exports);

        let mut code = CodeSection::new();
        for Func(_, refs) in &self.funcs {
            // `i32.const a; drop; ...; i32.const z; end`: refers to every symbol, returns the last.
            let mut func = Function::new([]);
            for (i, &symbol) in refs.iter().enumerate() {
                if i != 0 {
                    func.instruction(&Instruction::Drop);
                }
                let DataSymbol(_, offset, _) = self.symbols[symbol];
                func.raw(padded_i32_const(SEGMENT_BASE + offset as u32));
            }
            func.instruction(&Instruction::End);
            code.function(&func);
        }
        module.section(&code);

        let mut data = DataSection::new();
        let mut bytes = self.data.clone();
        for &(offset, symbol, ty, addend) in &self.data_relocs {
            let DataSymbol(_, target_offset, _) = self.symbols[symbol];
            let address = (SEGMENT_BASE + target_offset as u32).wrapping_add(addend as u32);
            let value = match ty {
                R_WASM_MEMORY_ADDR_I32 => address,
                R_WASM_MEMORY_ADDR_LOCREL_I32 => address.wrapping_sub(SEGMENT_BASE + offset as u32),
                _ => panic!("unsupported data relocation type {ty}"),
            };
            bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        data.active(0, &ConstExpr::i32_const(SEGMENT_BASE as i32), bytes);
        module.section(&data);

        module.finish()
    }

    fn build_wasm(&self) -> Vec<u8> {
        let mut wasm = self.build_module_prefix();
        let (code_section_index, immediates) = locate_code_immediates(&wasm);
        let (data_section_index, data_start) = locate_data_segment(&wasm);
        let reloc_symbols: Vec<u32> = self
            .funcs
            .iter()
            .flat_map(|Func(_, refs)| refs.iter().map(|&symbol| 1 + symbol as u32))
            .collect();
        assert_eq!(immediates.len(), reloc_symbols.len());

        // "linking" section, version 2
        let mut linking = vec![];
        uleb(2, &mut linking);
        {
            // WASM_SEGMENT_INFO
            let mut payload = vec![];
            uleb(1, &mut payload);
            name(".rodata", &mut payload);
            uleb(self.alignment, &mut payload); // alignment (log2)
            uleb(0, &mut payload); // flags
            linking.push(5);
            uleb(payload.len() as u32, &mut linking);
            linking.extend_from_slice(&payload);
        }
        {
            // WASM_SYMBOL_TABLE
            const SYMTAB_DATA: u8 = 1;
            const SYMTAB_TABLE: u8 = 5;
            const WASM_SYM_BINDING_LOCAL: u32 = 2;
            let mut payload = vec![];
            uleb(1 + self.symbols.len() as u32, &mut payload);
            // 0: the indirect function table
            payload.push(SYMTAB_TABLE);
            uleb(WASM_SYM_BINDING_LOCAL, &mut payload);
            uleb(0, &mut payload);
            name("__indirect_function_table", &mut payload);
            // 1..: data symbols, `(segment, offset, size)`
            for DataSymbol(symbol_name, offset, size) in &self.symbols {
                payload.push(SYMTAB_DATA);
                uleb(WASM_SYM_BINDING_LOCAL, &mut payload);
                name(symbol_name, &mut payload);
                uleb(0, &mut payload);
                uleb(*offset as u32, &mut payload);
                uleb(*size as u32, &mut payload);
            }
            linking.push(8);
            uleb(payload.len() as u32, &mut linking);
            linking.extend_from_slice(&payload);
        }
        wasm.push(0);
        CustomSection {
            name: Cow::Borrowed("linking"),
            data: Cow::Owned(linking),
        }
        .encode(&mut wasm);

        // wasm-split marker: [tag u8][payload_len uleb][payload]. (see wasm_split/../marker.rs)
        let ws_payload = [1u8, 1u8, 1u8];
        wasm.push(0);
        CustomSection {
            name: Cow::Borrowed(magic_constants::LINK_SECTION),
            data: Cow::Borrowed(&ws_payload),
        }
        .encode(&mut wasm);

        // "reloc.CODE": one entry per `i32.const` immediate
        let mut relocs = vec![];
        uleb(code_section_index, &mut relocs);
        uleb(immediates.len() as u32, &mut relocs);
        for (offset, symbol) in immediates.into_iter().zip(reloc_symbols) {
            relocs.push(R_WASM_MEMORY_ADDR_LEB);
            uleb(offset, &mut relocs);
            uleb(symbol, &mut relocs);
            relocs.push(0); // addend
        }
        wasm.push(0);
        CustomSection {
            name: Cow::Borrowed("reloc.CODE"),
            data: Cow::Owned(relocs),
        }
        .encode(&mut wasm);

        // "reloc.DATA": one entry per pointer inside the data
        if !self.data_relocs.is_empty() {
            let mut relocs = vec![];
            uleb(data_section_index, &mut relocs);
            uleb(self.data_relocs.len() as u32, &mut relocs);
            for &(offset, symbol, ty, addend) in &self.data_relocs {
                relocs.push(ty);
                uleb(data_start + offset as u32, &mut relocs);
                uleb(1 + symbol as u32, &mut relocs);
                sleb(addend, &mut relocs);
            }
            wasm.push(0);
            CustomSection {
                name: Cow::Borrowed("reloc.DATA"),
                data: Cow::Owned(relocs),
            }
            .encode(&mut wasm);
        }

        for payload in Parser::new(0).parse_all(&wasm) {
            payload.expect("input wasm is valid");
        }
        wasm
    }
}

/// `(section index of the code section, code relocation offsets of every
/// `i32.const` immediate, in function order)`.
fn locate_code_immediates(wasm: &[u8]) -> (u32, Vec<u32>) {
    let mut section_index: u32 = 0;
    let mut code_section: Option<(u32, u64)> = None;
    let mut immediates = vec![];
    for payload in Parser::new(0).parse_all(wasm) {
        let payload = payload.expect("valid wasm");
        if let Some((_, range)) = payload.as_section() {
            if let Payload::CodeSectionStart { .. } = payload {
                code_section = Some((section_index, range.start));
            }
            section_index += 1;
        }
        if let Payload::CodeSectionEntry(body) = payload {
            let (_, content_start) = code_section.expect("code section starts before entries");
            let mut ops = body.get_operators_reader().expect("operators");
            while !ops.eof() {
                let (op, offset) = ops.read_with_offset().expect("op");
                if let Operator::I32Const { .. } = op {
                    // the immediate follows the one-byte opcode
                    immediates.push((offset + 1 - content_start) as u32);
                }
            }
        }
    }
    let (index, _) = code_section.expect("code section");
    (index, immediates)
}

/// `(section index of the data section, data relocation offset of the first
/// byte of the only data segment)`.
fn locate_data_segment(wasm: &[u8]) -> (u32, u32) {
    let mut section_index: u32 = 0;
    for payload in Parser::new(0).parse_all(wasm) {
        let payload = payload.expect("valid wasm");
        let Some((_, range)) = payload.as_section() else {
            continue;
        };
        if let Payload::DataSection(reader) = &payload {
            let segment = reader
                .clone()
                .into_iter()
                .next()
                .expect("a data segment")
                .expect("valid data segment");
            let data_start = segment.range.end - segment.data.len() as u64;
            return (section_index, (data_start - range.start) as u32);
        }
        section_index += 1;
    }
    panic!("no data section");
}

/// The output modules of `transform` on an [`Input`].
struct Output {
    main: Vec<u8>,
    /// by split name
    splits: Vec<(String, Vec<u8>)>,
}

fn split(input: &Input) -> Output {
    let wasm = input.build_wasm();

    let mut tmp = tempfile::tempdir().expect("create tmpdir");
    tmp.disable_cleanup(true);
    let main_out = tmp.path().join("main.wasm");

    let mut opts = Options::new(&wasm);
    opts.output_dir = tmp.path();
    opts.main_out_path = &main_out;

    let output = transform(opts).expect("transform succeeds");

    let main = std::fs::read(&main_out).expect("read main.wasm output");
    let mut splits = vec![];
    for path in &output.split_modules {
        let file_name = path.file_name().unwrap().to_string_lossy();
        let bytes = std::fs::read(path).expect("read output module");
        if let Some(split) = file_name
            .strip_prefix("split_")
            .and_then(|rest| rest.strip_suffix(".wasm"))
        {
            splits.push((split.to_string(), bytes));
        } else if file_name.starts_with("chunk_") {
            // chunks hold data shared between splits, of which there are none here
        } else {
            panic!("unexpected output module {file_name}");
        }
    }

    tmp.disable_cleanup(false);
    Output { main, splits }
}

impl Output {
    fn split(&self, name: &str) -> &[u8] {
        &self
            .splits
            .iter()
            .find(|(split, _)| split == name)
            .unwrap_or_else(|| panic!("no split module {name:?}"))
            .1
    }
}

/// The `(address, bytes)` of every non-empty active data segment.
fn data_segments(wasm: &[u8]) -> Vec<(u32, Vec<u8>)> {
    let mut segments = vec![];
    for payload in Parser::new(0).parse_all(wasm) {
        let Payload::DataSection(reader) = payload.expect("valid wasm") else {
            continue;
        };
        for segment in reader {
            let segment = segment.expect("valid data segment");
            if segment.data.is_empty() {
                continue;
            }
            let DataKind::Active { offset_expr, .. } = segment.kind else {
                panic!("expected active data segments only");
            };
            let Operator::I32Const { value } = offset_expr
                .get_operators_reader()
                .read()
                .expect("offset expression")
            else {
                panic!("expected a constant segment offset");
            };
            segments.push((value as u32, segment.data.to_vec()));
        }
    }
    segments
}

/// The one non-empty data segment of a module.
fn single_data_segment(wasm: &[u8]) -> (u32, Vec<u8>) {
    let mut segments = data_segments(wasm);
    assert_eq!(
        segments.len(),
        1,
        "expected exactly one non-empty data segment, got {segments:?}"
    );
    segments.pop().unwrap()
}

/// The `i32.const` immediates of every defined function, in function order.
fn function_constants(wasm: &[u8]) -> Vec<Vec<u32>> {
    let mut functions = vec![];
    for payload in Parser::new(0).parse_all(wasm) {
        let Payload::CodeSectionEntry(body) = payload.expect("valid wasm") else {
            continue;
        };
        let mut constants = vec![];
        let mut ops = body.get_operators_reader().expect("operators");
        while !ops.eof() {
            if let Operator::I32Const { value } = ops.read().expect("op") {
                constants.push(value as u32);
            }
        }
        functions.push(constants);
    }
    functions
}

/// The `i32.const` immediates of the defined function exported as `export_name`.
fn exported_function_constants(wasm: &[u8], export_name: &str) -> Vec<u32> {
    let mut import_count = 0;
    let mut target = None;
    for payload in Parser::new(0).parse_all(wasm) {
        match payload.expect("valid wasm") {
            Payload::ImportSection(reader) => {
                for import in reader {
                    let Ok(wasmparser::Imports::Single(_, import)) = import else {
                        panic!("valid import")
                    };
                    if matches!(import.ty, wasmparser::TypeRef::Func(_)) {
                        import_count += 1;
                    }
                }
            }
            Payload::ExportSection(reader) => {
                for export in reader {
                    let export = export.expect("valid export");
                    if export.name == export_name && export.kind == wasmparser::ExternalKind::Func {
                        target = Some(export.index as usize);
                    }
                }
            }
            _ => {}
        }
    }
    let index = target.unwrap_or_else(|| panic!("export {export_name:?} not found"));
    function_constants(wasm).swap_remove(index - import_count)
}

/// Split modules export nothing; their entry is reached through the function
/// table. Asserts that some function of `wasm` has exactly these constants.
fn assert_some_function_refers_to(wasm: &[u8], constants: &[u32]) {
    let functions = function_constants(wasm);
    assert!(
        functions.contains(&constants.to_vec()),
        "expected a function referring to {constants:?}, got {functions:?}",
    );
}

#[test]
fn location_relative_pointers_are_relocated_relative_to_their_new_location() {
    let _ = tracing_subscriber::fmt::try_init();

    // table = [relative pointer][absolute pointer], both to `target`
    let mut data = vec![0; 8];
    data.extend_from_slice(b"XXXX");
    let input = Input {
        data: data.clone(),
        alignment: 2,
        symbols: vec![DataSymbol("table", 0, 8), DataSymbol("target", 8, 4)],
        funcs: vec![
            Func(Owner::Main("main_reads"), vec![1]),
            Func(Owner::Split("a"), vec![0]),
        ],
        data_relocs: vec![
            (0, 1, R_WASM_MEMORY_ADDR_LOCREL_I32, 0),
            (4, 1, R_WASM_MEMORY_ADDR_I32, 0),
        ],
    };
    let output = split(&input);

    let (target_addr, target_bytes) = single_data_segment(&output.main);
    assert_eq!(target_bytes, b"XXXX");
    assert_eq!(
        exported_function_constants(&output.main, "main_reads"),
        vec![target_addr]
    );

    let split_a = output.split("a");
    let (table_addr, table_bytes) = single_data_segment(split_a);
    assert_ne!(table_addr, SEGMENT_BASE, "the table should have moved");
    assert_some_function_refers_to(split_a, &[table_addr]);
    let relative = i32::from_le_bytes(table_bytes[0..4].try_into().unwrap());
    let absolute = u32::from_le_bytes(table_bytes[4..8].try_into().unwrap());
    assert_eq!(absolute, target_addr, "absolute pointer");
    assert_eq!(
        relative,
        target_addr as i32 - table_addr as i32,
        "relative pointer must be relative to its own new address",
    );
}

#[test]
fn location_relative_pointer_to_an_unmoved_target_is_still_updated() {
    let _ = tracing_subscriber::fmt::try_init();

    // The pointer (split a) refers to `end`, a zero-sized symbol like `__heap_base`, which is
    // not relocated. `word` (main) makes the pointer move.
    let mut data = vec![0; 4];
    data.extend_from_slice(b"WWWW");
    let input = Input {
        data: data.clone(),
        alignment: 2,
        symbols: vec![
            DataSymbol("pointer", 0, 4),
            DataSymbol("word", 4, 4),
            DataSymbol("end", 8, 0),
        ],
        funcs: vec![
            Func(Owner::Main("main_reads"), vec![1]),
            Func(Owner::Split("a"), vec![0]),
        ],
        data_relocs: vec![(0, 2, R_WASM_MEMORY_ADDR_LOCREL_I32, 0)],
    };
    let output = split(&input);

    let end_addr = SEGMENT_BASE + 8;
    let (word_addr, word_bytes) = single_data_segment(&output.main);
    assert_eq!(word_bytes, b"WWWW");
    let (pointer_addr, pointer_bytes) = single_data_segment(output.split("a"));
    assert_ne!(pointer_addr, SEGMENT_BASE, "the pointer should have moved");
    assert!(pointer_addr + 4 <= end_addr && word_addr + 4 <= end_addr);
    let relative = i32::from_le_bytes(pointer_bytes[0..4].try_into().unwrap());
    assert_eq!(
        relative,
        end_addr as i32 - pointer_addr as i32,
        "relative pointer must be updated although its target did not move",
    );
}

#[test]
fn location_relative_pointer_keeps_the_low_32_bits() {
    let _ = tracing_subscriber::fmt::try_init();

    // The pointer (main) refers to its target, which pulls the target into main right after
    // it, so `S + A - P` with the largest addend exceeds `i32::MAX`.
    let mut data = vec![0; 4];
    data.extend_from_slice(b"XXXX");
    let input = Input {
        data: data.clone(),
        alignment: 2,
        symbols: vec![DataSymbol("pointer", 0, 4), DataSymbol("target", 4, 4)],
        funcs: vec![
            Func(Owner::Main("main_reads"), vec![0]),
            Func(Owner::Split("a"), vec![1]),
        ],
        data_relocs: vec![(0, 1, R_WASM_MEMORY_ADDR_LOCREL_I32, i32::MAX)],
    };
    let output = split(&input);

    let (pointer_addr, main_bytes) = single_data_segment(&output.main);
    assert_eq!(
        &main_bytes[4..8],
        b"XXXX",
        "the target follows the pointer in main"
    );
    assert_eq!(data_segments(output.split("a")), vec![]);
    let target_addr = pointer_addr + 4;
    let value = u32::from_le_bytes(main_bytes[0..4].try_into().unwrap());
    let expected = target_addr
        .wrapping_add(i32::MAX as u32)
        .wrapping_sub(pointer_addr);
    assert_eq!(value, expected, "low 32 bits of S + A - P");
}
