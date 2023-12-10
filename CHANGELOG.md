## [0.3.0] - 2023-12-10

### Bug fixes
* Fix some typos, add dev container, faux streaming [#200](https://github.com/pinecone-io/canopy/pull/200) (Thanks @eburnette!)
* CLI requires OpenAI API key, even if OpenAI is not being used by[#208](https://github.com/pinecone-io/canopy/pull/208)
* CLI: read config file from env location[#190](https://github.com/pinecone-io/canopy/pull/190) (Thanks @MichaelAnckaert!)


### Documentation
* Add document field explanations and python version badges [#187](https://github.com/pinecone-io/canopy/pull/187)
* Update README.md [#192](https://github.com/pinecone-io/canopy/pull/192) (Thanks @tomer-w!)
* Tweaks to CLI help texts [#193](https://github.com/pinecone-io/canopy/pull/193) (Thanks @jseldess!)
* Update README.md and change href [#202](https://github.com/pinecone-io/canopy/pull/202)

### CI Improvements
* Added bug-report template [#184](https://github.com/pinecone-io/canopy/pull/184)
* Add feature-request.yml [#209](https://github.com/pinecone-io/canopy/pull/209)

### Added
* Add Anyscale Endpoint support and Llama Tokenizer [#173](https://github.com/pinecone-io/canopy/pull/173) (Thanks @kylehh!)
* Add last message query generator [#210](https://github.com/pinecone-io/canopy/pull/210)


**Full Changelog**: https://github.com/pinecone-io/canopy/compare/V0.2.0...V0.3.0

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