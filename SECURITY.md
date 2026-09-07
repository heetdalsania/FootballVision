# Security policy

## Supported versions

FootballVision is currently developed on the `main` branch. Security fixes are
applied there; older commits and unmaintained forks are not supported releases.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Use the
repository's private
[security advisory form](https://github.com/heetdalsania/FootballVision/security/advisories/new)
and include:

- the affected component and commit;
- reproduction steps or a minimal proof of concept;
- the likely impact; and
- any suggested mitigation.

Do not include real match footage, API keys, private databases, or personal
information. You should receive an initial response within seven days. A fix
and disclosure timeline will be coordinated after the report is reproduced.

## Scope notes

FootballVision binds to `127.0.0.1` by default and is designed for trusted local
use. Binding the server to a public interface, exposing it through a tunnel, or
processing untrusted files changes its threat model and is not a supported
deployment configuration.
