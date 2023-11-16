## [0.2.0] - 2023-11-15

### Bug fixes
- Bug fix in E2E test that prevented running `pytest tests/` [#175](https://github.com/pinecone-io/canopy/pull/175)

### Added
- Upgrade openai client dependency to v.1.2.3 (required code change) [#171](https://github.com/pinecone-io/canopy/pull/171), [#178](https://github.com/pinecone-io/canopy/pull/178)

### Breaking changes
- Added versioning to Canopy server's API [#169](https://github.com/pinecone-io/canopy/pull/169)

**Full Changelog**: https://github.com/pinecone-io/canopy/compare/V0.1.4...V0.2.0
## [0.1.4] - 2023-11-14

### Bug fixes

- Fixed error when trying to run `canopy chat` on Windows [#166](https://github.com/pinecone-io/canopy/issues/166)
- Fixed `canopy stop` on Windows [#166](https://github.com/pinecone-io/canopy/issues/166#issuecomment-1805894866)
- Update incorrect pinecone quick start path [#168](https://github.com/pinecone-io/canopy/pull/168) (Thanks @abpai!)


## [0.1.3] - 2023-11-09
- Edit description on pyproject.toml.

## [0.1.2] - 2023-11-09

- Added the ability to load individual text files from a directory
- Bumped the `pinecone-text` dependency to fix a numpy dependency issue

## [0.1.1] - 2023-11-07

- Readme fixes

## [0.1.0] - 2023-11-05

- Initial release