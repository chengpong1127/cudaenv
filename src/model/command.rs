use std::ffi::OsStr;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommandSpec {
    pub program: String,
    pub args: Vec<String>,
}

impl CommandSpec {
    pub fn new<I, S>(program: &str, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        Self {
            program: program.to_owned(),
            args: args
                .into_iter()
                .map(|arg| arg.as_ref().to_string_lossy().into_owned())
                .collect(),
        }
    }

    pub fn sudo<I, S>(program: &str, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        let mut all = vec![program.to_owned()];
        all.extend(
            args.into_iter()
                .map(|arg| arg.as_ref().to_string_lossy().into_owned()),
        );
        Self {
            program: "sudo".to_owned(),
            args: all,
        }
    }

    pub fn display(&self) -> String {
        std::iter::once(self.program.as_str())
            .chain(self.args.iter().map(String::as_str))
            .map(quote)
            .collect::<Vec<_>>()
            .join(" ")
    }
}

fn quote(value: &str) -> String {
    if value
        .bytes()
        .all(|c| c.is_ascii_alphanumeric() || b"-_=+./:".contains(&c))
    {
        value.to_owned()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safely_quotes_display_arguments() {
        let command = CommandSpec::new("tool", ["plain", "two words", "it's"]);
        assert_eq!(command.display(), "tool plain 'two words' 'it'\\''s'");
    }

    #[test]
    fn quotes_package_wildcards_for_safe_copying() {
        let command = CommandSpec::sudo("apt", ["remove", "cuda-toolkit*"]);
        assert_eq!(command.display(), "sudo apt remove 'cuda-toolkit*'");
    }
}
