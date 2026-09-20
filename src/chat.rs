//! `chat_template.jinja`, hand-compiled.
//!
//! The artifact ships a 150-line jinja template; rather than write a jinja
//! engine, this is that template read top to bottom and written out as Rust.
//! It follows the source statement by statement, including the parts that do
//! nothing (`ns.last_query_index` is computed and never read) and the parts
//! that look like bugs (a leading `tool` message emits no `<|im_start|>user`,
//! because `loop.previtem` is undefined on the first iteration).
//!
//! What the template does, in order:
//!
//!   1. if `tools` is a non-mapping iterable, a `system` block listing them
//!      -- each one `tojson`'d, then the calling convention, then the system
//!      message's own content if there is one; otherwise, if `messages[0]`
//!      is a system message, that as the system block,
//!   2. a backwards scan for the last user message that is not a bare
//!      `<tool_response>`, which only ever raises -- the index it records is
//!      never used,
//!   3. the messages, each rendered by role,
//!   4. `<|im_start|>assistant\n` and then either `<think>\n\n</think>\n\n`
//!      (thinking off) or `<think>\n` (thinking on).
//!
//! Every `{{ content }}` in the template goes through `render_content(...)
//! |trim`, and jinja's `trim` is Python's `str.strip()`, so the whitespace it
//! strips is `str.isspace`, a slightly wider set than the regex's `\s`.
//!
//! The two blocks that render through jinja's `tojson` -- the tool schemas
//! and an assistant's `tool_calls` arguments -- are what `json::Value::to_json`
//! exists for: `transformers` replaces that filter with
//! `json.dumps(..., ensure_ascii=False)`, so both want a JSON writer over a
//! value that keeps its key order. Nothing else in this port does.
//!
//! Still not here: there is no *parser* for what the model writes back. The
//! template tells it to answer in `<tool_call><function=...>` XML rather than
//! JSON, and reading that back into a `ToolCall` is a caller's job.

use crate::json::{Map, Value};
use crate::tokenizer::tables::PY_WHITESPACE;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// One item of a list-shaped `content`, as `render_content` tests for them.
#[derive(Debug, Clone)]
pub enum Part {
    Image,
    Video,
    Text(String),
}

#[derive(Debug, Clone)]
pub enum Content {
    /// `content is string`
    Text(String),
    /// `content is iterable and content is not mapping`
    Parts(Vec<Part>),
    /// `content is none or content is undefined`
    None,
}

/// One entry of `message.tool_calls`, already unwrapped: the template reads
/// `tool_call.function` when it is there and the bare object when it is not,
/// and both shapes end up here.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub name: String,
    /// `tool_call.arguments`, walked with jinja's `items` filter, so in
    /// source order. An absent `arguments` and an empty one render the same
    /// -- no `<parameter=...>` blocks at all -- so both are an empty `Map`.
    pub arguments: Map,
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: Content,
    /// `message.reasoning_content`, which takes precedence over a `<think>`
    /// block inside the content.
    pub reasoning_content: Option<String>,
    /// `message.tool_calls`, on an assistant message. Every other role
    /// ignores it, as the template does.
    pub tool_calls: Vec<ToolCall>,
}

impl Message {
    pub fn text(role: Role, s: &str) -> Message {
        Message {
            role,
            content: Content::Text(s.to_string()),
            reasoning_content: None,
            tool_calls: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub enum TemplateError {
    NoMessages,
    NoUserQuery,
    SystemNotFirst,
    ImageInSystem,
    VideoInSystem,
}

impl std::fmt::Display for TemplateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            TemplateError::NoMessages => "No messages provided.",
            TemplateError::NoUserQuery => "No user query found in messages.",
            TemplateError::SystemNotFirst => "System message must be at the beginning.",
            TemplateError::ImageInSystem => "System message cannot contain images.",
            TemplateError::VideoInSystem => "System message cannot contain videos.",
        };
        f.write_str(s)
    }
}

fn is_py_space(c: char) -> bool {
    let c = c as u32;
    PY_WHITESPACE
        .binary_search_by(|&(lo, hi)| {
            if c < lo {
                std::cmp::Ordering::Greater
            } else if c > hi {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// Python's `str.strip()`, which is what jinja's `|trim` calls.
fn trim(s: &str) -> &str {
    s.trim_matches(is_py_space)
}

/// `render_content(content, do_vision_count, is_system_content)`. The vision
/// counters only feed `add_vision_id`, which this template is never called
/// with, so they are not kept.
fn render_content(content: &Content, is_system: bool) -> Result<String, TemplateError> {
    match content {
        Content::Text(s) => Ok(s.clone()),
        Content::Parts(parts) => {
            let mut out = String::new();
            for p in parts {
                match p {
                    Part::Image => {
                        if is_system {
                            return Err(TemplateError::ImageInSystem);
                        }
                        out.push_str("<|vision_start|><|image_pad|><|vision_end|>");
                    }
                    Part::Video => {
                        if is_system {
                            return Err(TemplateError::VideoInSystem);
                        }
                        out.push_str("<|vision_start|><|video_pad|><|vision_end|>");
                    }
                    Part::Text(t) => out.push_str(t),
                }
            }
            Ok(out)
        }
        Content::None => Ok(String::new()),
    }
}

fn rendered(content: &Content, is_system: bool) -> Result<String, TemplateError> {
    Ok(trim(&render_content(content, is_system)?).to_string())
}

/// The calling convention the tools block spells out, verbatim.
const TOOL_CALL_INSTRUCTIONS: &str = "\n\n\
     If you choose to call a function ONLY reply in the following format with NO suffix:\n\n\
     <tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n\
     value_1\n</parameter>\n<parameter=example_parameter_2>\n\
     This is the value for the second parameter\nthat can span\nmultiple lines\n\
     </parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n\
     - Function calls MUST follow the specified format: an inner <function=...></function> \
     block must be nested within <tool_call></tool_call> XML tags\n\
     - Required parameters MUST be specified\n\
     - You may provide optional reasoning for your function call in natural language BEFORE \
     the function call, but NOT after\n\
     - If there is no function call available, answer the question like normal with your \
     current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// `tokenizer.apply_chat_template(messages, tools, add_generation_prompt,
/// enable_thinking)`. An empty `tools` is jinja's falsy one.
pub fn apply_chat_template(
    messages: &[Message],
    tools: &[Value],
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> Result<String, TemplateError> {
    if messages.is_empty() {
        return Err(TemplateError::NoMessages);
    }
    let mut out = String::new();

    if !tools.is_empty() {
        out.push_str("<|im_start|>system\n");
        out.push_str("# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for tool in tools {
            out.push('\n');
            out.push_str(&tool.to_json());
        }
        out.push_str("\n</tools>");
        out.push_str(TOOL_CALL_INSTRUCTIONS);
        // Unlike the `else` below, an empty system message adds nothing here
        // -- not even the two newlines.
        if messages[0].role == Role::System {
            let content = rendered(&messages[0].content, true)?;
            if !content.is_empty() {
                out.push_str("\n\n");
                out.push_str(&content);
            }
        }
        out.push_str("<|im_end|>\n");
    } else if messages[0].role == Role::System {
        let content = rendered(&messages[0].content, true)?;
        out.push_str("<|im_start|>system\n");
        out.push_str(&content);
        out.push_str("<|im_end|>\n");
    }

    // The backwards scan. `ns.last_query_index` is dead in the template, so
    // only the flag it also sets survives here.
    let mut multi_step_tool = true;
    for message in messages.iter().rev() {
        if multi_step_tool && message.role == Role::User {
            let content = rendered(&message.content, false)?;
            if !(content.starts_with("<tool_response>") && content.ends_with("</tool_response>")) {
                multi_step_tool = false;
            }
        }
    }
    if multi_step_tool {
        return Err(TemplateError::NoUserQuery);
    }

    for (i, message) in messages.iter().enumerate() {
        let mut content = rendered(&message.content, false)?;
        match message.role {
            Role::System => {
                if i != 0 {
                    return Err(TemplateError::SystemNotFirst);
                }
                // The system block was already emitted above.
            }
            Role::User => {
                out.push_str("<|im_start|>user\n");
                out.push_str(&content);
                out.push_str("<|im_end|>\n");
            }
            Role::Assistant => {
                let reasoning = match &message.reasoning_content {
                    Some(r) => r.clone(),
                    None => {
                        if content.contains("</think>") {
                            let head = content.split("</think>").next().unwrap().to_string();
                            let r = head
                                .trim_end_matches('\n')
                                .rsplit("<think>")
                                .next()
                                .unwrap()
                                .trim_start_matches('\n')
                                .to_string();
                            content = content
                                .rsplit("</think>")
                                .next()
                                .unwrap()
                                .trim_start_matches('\n')
                                .to_string();
                            r
                        } else {
                            String::new()
                        }
                    }
                };
                out.push_str("<|im_start|>assistant\n<think>\n");
                out.push_str(trim(&reasoning));
                out.push_str("\n</think>\n\n");
                out.push_str(&content);
                for (j, call) in message.tool_calls.iter().enumerate() {
                    // The separator is the only thing the position changes:
                    // a blank line after content that survived `|trim`, none
                    // after content that did not, one newline between calls.
                    if j > 0 {
                        out.push('\n');
                    } else if !trim(&content).is_empty() {
                        out.push_str("\n\n");
                    }
                    out.push_str("<tool_call>\n<function=");
                    out.push_str(&call.name);
                    out.push_str(">\n");
                    for (name, value) in &call.arguments {
                        out.push_str("<parameter=");
                        out.push_str(name);
                        out.push_str(">\n");
                        // `args_value | string if args_value is string else
                        // args_value | tojson`: a string goes in raw, without
                        // its quotes, and everything else as JSON.
                        match value {
                            Value::Str(s) => out.push_str(s),
                            v => out.push_str(&v.to_json()),
                        }
                        out.push_str("\n</parameter>\n");
                    }
                    out.push_str("</function>\n</tool_call>");
                }
                out.push_str("<|im_end|>\n");
            }
            Role::Tool => {
                if i > 0 && messages[i - 1].role != Role::Tool {
                    out.push_str("<|im_start|>user");
                }
                out.push_str("\n<tool_response>\n");
                out.push_str(&content);
                out.push_str("\n</tool_response>");
                let last = i + 1 == messages.len();
                if last || messages[i + 1].role != Role::Tool {
                    out.push_str("<|im_end|>\n");
                }
            }
        }
    }

    if add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
        if enable_thinking {
            out.push_str("<think>\n");
        } else {
            out.push_str("<think>\n\n</think>\n\n");
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plain_turn() {
        let msgs = [Message::text(Role::User, "hi")];
        assert_eq!(
            apply_chat_template(&msgs, &[], true, false).unwrap(),
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        assert_eq!(
            apply_chat_template(&msgs, &[], true, true).unwrap(),
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
    }

    #[test]
    fn no_user_message_is_an_error() {
        let msgs = [Message::text(Role::System, "be nice")];
        assert!(matches!(
            apply_chat_template(&msgs, &[], true, false),
            Err(TemplateError::NoUserQuery)
        ));
    }
}
