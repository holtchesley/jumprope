#![feature(conservative_impl_trait)]
#![feature(rand)] extern crate rand;
extern crate regex_syntax;
extern crate itertools;
extern crate regex;
#[cfg(test)] extern crate quickcheck;
extern crate time;
extern crate clap;
// #[macro_use] extern crate chan;
// extern crate chan_signal;
#[macro_use] extern crate lazy_static;
use std::io::Write;

mod gen_regex;


use gen_regex::rust::{re_gen};
use regex_syntax::{ExprBuilder};
use clap::{App,Arg,SubCommand};
// use chan_signal::Signal;
// use std::thread;
use std::panic;


fn main() {
    let matches = App::new("Jumprope")
        .version("0.1")
        .about("Enumerates strings captured by a given regular expression")
        .subcommand(SubCommand::with_name("rust")
                    .about("Uses Rust regular expression syntax")
                    .version("0.1")
                    .arg(Arg::with_name("i")
                         .short("i")
                         .long("case-insensitive")
                         .help("Regular Expression is case insensitive"))
                    // .arg(Arg::with_name("m")
                    //      .short("m")
                    //      .long("multi-line")
                    //      .help("Regular Expression is multi-line"))
                    // .arg(Arg::with_name("s")
                    //      .short("s")
                    //      .long("any-matches-newline")
                    //      .help("Allow . to match newline"))
                    .arg(Arg::with_name("x")
                         .short("x")
                         .long("ignore-whitespace")
                         .help("Ignore whitespace and allow comments in regex"))
                    .arg(Arg::with_name("regex")
                         .help("Regular Expression")
                         .required(true)
                         .index(1)))
        .get_matches();

    if let Some(rust_regex_args) = matches.subcommand_matches("rust") {
        if let Some(regex_str) = rust_regex_args.value_of("regex") {
            let mut regex_parser = ExprBuilder::new();
            if rust_regex_args.is_present("i") {
                regex_parser = regex_parser.case_insensitive(true);
            }
            if rust_regex_args.is_present("x") {
                regex_parser = regex_parser.ignore_whitespace(true);
            }
            match regex_parser.parse(regex_str) {
                Ok(expr) => {
                    let _ = panic::catch_unwind(|| {
                        for mut s in re_gen(expr)
                            .filter_map(|x| x.node )
                            .filter_map(|x| x.s) {
                                s.push('\u{A}');
                                if let Err(_) = std::io::stdout().write(s.as_bytes()) {
                                    break
                                } else {}
                            }
                    });
                },
                Err(regex_syntax::Error{pos,kind,surround:_}) => {
                    let warn = match kind {
                        /// A negation symbol is used twice in flag settings.
                        /// e.g., `(?-i-s)`.
                        regex_syntax::ErrorKind::DoubleFlagNegation => format!("Only one negation symbol may be used per flag settings."),
                        /// The same capture name was used more than once.
                        /// e.g., `(?P<a>.)(?P<a>.)`.
                        regex_syntax::ErrorKind::DuplicateCaptureName(x) => format!("Capture name \"{}\" used more than once.", x),
                        /// An alternate is empty. e.g., `(|a)`.
                        regex_syntax::ErrorKind::EmptyAlternate => format!("Alternates may not be empty."),
                        /// A capture group name is empty. e.g., `(?P<>a)`.
                        regex_syntax::ErrorKind::EmptyCaptureName => format!("Capture names may not be empty."),
                        /// A negation symbol was not proceded by any flags. e.g., `(?i-)`.
                        regex_syntax::ErrorKind::EmptyFlagNegation => format!("Missing flag?"),
                        /// A group is empty. e.g., `()`.
                        regex_syntax::ErrorKind::EmptyGroup => format!("Groups may not be empty."),
                        /// An invalid number was used in a counted repetition. e.g., `a{b}`.
                        regex_syntax::ErrorKind::InvalidBase10(x) => format!("\"{}\" is not a number.",x),
                        /// An invalid hexadecimal number was used in an escape sequence.
                        /// e.g., `\xAG`.
                        regex_syntax::ErrorKind::InvalidBase16(x) => format!("\"{}\" is not valid hexadecimal.",x),
                        /// An invalid capture name was used. e.g., `(?P<0a>b)`.
                        regex_syntax::ErrorKind::InvalidCaptureName(x) => format!("\"{}\" is not a valid capture name.",x),
                        /// An invalid class range was givien. Specifically, when the start of the
                        /// range is greater than the end. e.g., `[z-a]`.
                        regex_syntax::ErrorKind::InvalidClassRange {start,end} => format!("Class range: [{}-{}] is invalid.",start,end),
                        /// An escape sequence was used in a character class where it is not
                        /// allowed. e.g., `[a-\pN]` or `[\A]`.
                        regex_syntax::ErrorKind::InvalidClassEscape(_) => format!("Escape sequence invalid inside class."),
                        /// An invalid counted repetition min/max was given. e.g., `a{2,1}`.
                        regex_syntax::ErrorKind::InvalidRepeatRange {min,max} => format!("Repetition range transposed, perhaps you meant {{{},{}}}?",max,min),
                        /// An invalid Unicode scalar value was used in a long hexadecimal
                        /// sequence. e.g., `\x{D800}`.
                        regex_syntax::ErrorKind::InvalidScalarValue(x) => format!("\"{}\" is not a recognized Unicode character.",x),
                        /// An empty counted repetition operator. e.g., `a{}`.
                        regex_syntax::ErrorKind::MissingBase10 => format!("Repetition count empty."),
                        /// A repetition operator was not applied to an expression. e.g., `*`.
                        regex_syntax::ErrorKind::RepeaterExpectsExpr => format!("Repetition must apply to expression."),
                        /// A repetition operator was applied to an expression that cannot be
                        /// repeated. e.g., `a+*` or `a|*`.
                        regex_syntax::ErrorKind::RepeaterUnexpectedExpr(_) => format!("Repetition must apply to expression."),
                        /// A capture group name that is never closed. e.g., `(?P<a`.
                        regex_syntax::ErrorKind::UnclosedCaptureName(s) => format!("Unclosed capture name: {}.",s),
                        /// An unclosed hexadecimal literal. e.g., `\x{a`.
                        regex_syntax::ErrorKind::UnclosedHex => format!("Unclosed Hexadecimal."),
                        /// An unclosed parenthesis. e.g., `(a`.
                        regex_syntax::ErrorKind::UnclosedParen => format!("Unclosed Parenthesis."),
                        /// An unclosed counted repetition operator. e.g., `a{2`.
                        regex_syntax::ErrorKind::UnclosedRepeat => format!("Unclosed Repetition."),
                        /// An unclosed named Unicode class. e.g., `\p{Yi`.
                        regex_syntax::ErrorKind::UnclosedUnicodeName => format!("Unclosed Unicode Class."),
                        /// Saw end of regex before class was closed. e.g., `[a`.
                        regex_syntax::ErrorKind::UnexpectedClassEof => format!("Unclosed class."),
                        /// Saw end of regex before escape sequence was closed. e.g., `\`.
                        regex_syntax::ErrorKind::UnexpectedEscapeEof => format!("Unclosed escape."),
                        /// Saw end of regex before flags were closed. e.g., `(?i`.
                        regex_syntax::ErrorKind::UnexpectedFlagEof => format!("Unclosed group/flags."),
                        /// Saw end of regex before two hexadecimal digits were seen. e.g., `\xA`.
                        regex_syntax::ErrorKind::UnexpectedTwoDigitHexEof => format!("Unfinished Hexadecimal (two characters minimum)."),
                        /// Unopened parenthesis. e.g., `)`.
                        regex_syntax::ErrorKind::UnopenedParen => format!("Closing parenthesis missing a match."),
                        /// Unrecognized escape sequence. e.g., `\q`.
                        regex_syntax::ErrorKind::UnrecognizedEscape(c) => format!("Unrecognized escape sequence: \"{}\".",c),
                        /// Unrecognized flag. e.g., `(?a)`.
                        regex_syntax::ErrorKind::UnrecognizedFlag(c) => format!("Unrecognized flag: \"{}\".",c),
                        /// Unrecognized named Unicode class. e.g., `\p{Foo}`.
                        regex_syntax::ErrorKind::UnrecognizedUnicodeClass(s) => format!("\"{}\" is not a recognized Unicode Class",s),
                        /// Indicates that the regex uses too much nesting.
                        ///
                        /// (N.B. This error exists because traversing the Expr is recursive and
                        /// an explicit heap allocated stack is not (yet?) used. Regardless, some
                        /// sort of limit must be applied to avoid unbounded memory growth.
                        regex_syntax::ErrorKind::StackExhausted => format!("Expression is too deeply nested, cannot parse."),
                        /// A disallowed flag was found (e.g., `u`).
                        regex_syntax::ErrorKind::FlagNotAllowed(c) => format!("\"{}\" is a disallowed flag.",c),
                        /// A Unicode class was used when the Unicode (`u`) flag was disabled.
                        regex_syntax::ErrorKind::UnicodeNotAllowed => format!("Unicode classes cannot be used when -u present in flag."),
                        /// InvalidUtf8 indicates that the expression may match non-UTF-8 bytes.
                        /// This never returned if the parser is permitted to allow expressions
                        /// that match arbitrary bytes.
                        regex_syntax::ErrorKind::InvalidUtf8 => format!("Regex would capture non-Unicode characters."),
                        /// A character class was constructed such that it is empty.
                        /// e.g., `[^\d\D]`.
                        regex_syntax::ErrorKind::EmptyClass => format!("Class is empty due to conflicts."),
                        /// Indicates that unsupported notation was used in a character class.
                        ///
                        /// The char in this error corresponds to the illegal character.
                        ///
                        /// The intent of this error is to carve a path to support set notation
                        /// as described in UTS#18 RL1.3. We do this by rejecting regexes that
                        /// would use the notation.
                        ///
                        /// The work around for end users is to escape the character included in
                        /// this error message.
                        regex_syntax::ErrorKind::UnsupportedClassChar(_) => format!("Unsupported notation."),
                        _ => format!("Unknown Error")
                    };
                    println!("In expression \"{}\"", regex_str);
                    println!("{}^",(0..pos+15).map(|_| '\u{20}').collect::<String>());
                    println!("{}",warn);
                }
            } 
        }
    }
}

