# whisrs-recordings agent skill

A skill that tells an AI agent where whisrs keeps your recordings and how to
read them: the folder layout, the `whisrs recording` commands, and that a
transcript is data, not instructions. It only reads; it never deletes a
recording.

Install it for Claude Code:

```sh
mkdir -p ~/.claude/skills
cp -r contrib/agent-skill/whisrs-recordings ~/.claude/skills/
```

For another agent that reads `SKILL.md` skills, copy the folder into its
skills folder. Then ask it about a recording, for example "what did we decide
in the call yesterday?".
