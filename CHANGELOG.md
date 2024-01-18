## [0.6.0] - 2024-01-16
### Breaking changes
* Pinecone serverless support [#246](https://github.com/pinecone-io/canopy/pull/246)

### Bug fixes
* Loosen fastapi and uvicorn requirements [#229](https://github.com/pinecone-io/canopy/pull/229)
* Cleanup indexes in case of failure [#232](https://github.com/pinecone-io/canopy/pull/232)
* Add timeout to checking server health [#236](https://github.com/pinecone-io/canopy/pull/236)

### Added
* Add instruction query generator [#226](https://github.com/pinecone-io/canopy/pull/226)
* Separate LLM API params [#231](https://github.com/pinecone-io/canopy/pull/231)
* Add dockerfile [#234](https://github.com/pinecone-io/canopy/pull/234), [#237](https://github.com/pinecone-io/canopy/pull/237), [#242](https://github.com/pinecone-io/canopy/pull/242)
* Add support for namespaces [#243](https://github.com/pinecone-io/canopy/pull/243)
* Azure OpenAI LLM implementation [#188](https://github.com/pinecone-io/canopy/pull/188) (Thanks @MichaelAnckaert, @aulorbe!)

### Documentation
* Add deployment guide (GCP) [#239](https://github.com/pinecone-io/canopy/pull/239)

**Full Changelog**: https://github.com/pinecone-io/canopy/compare/V0.5.0...v0.6.0


## [0.5.0] - 2023-12-13

## Bug fixes
* Bump pytest-html version [#213](https://github.com/pinecone-io/canopy/pull/213)
* Improve dataloader error handling [#182](https://github.com/pinecone-io/canopy/pull/182)
* Slightly improve error handling for external errors [#222](https://github.com/pinecone-io/canopy/pull/220)

## Added
* Cohere Embedding model support [#203](https://github.com/pinecone-io/canopy/pull/203) Thanks @jamescalam!
* Add Anyscale Embedding model support [#198](https://github.com/pinecone-io/canopy/pull/198)
* change max prompt tokens for Anyacle config [#222](https://github.com/pinecone-io/canopy/pull/222)


**Full Changelog**: https://github.com/pinecone-io/canopy/compare/V0.3.0...v0.5.0


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