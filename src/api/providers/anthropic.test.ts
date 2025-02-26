import { describe, it } from "mocha"
import "should"
import { AnthropicHandler } from "./anthropic"
import type { Anthropic } from "@anthropic-ai/sdk"

describe("AnthropicHandler", () => {
	describe("createMessage", () => {
		async function* dummyStream() {
			yield { type: "message_start", message: { usage: { input_tokens: 123, output_tokens: 456 } } }
		}

		async function* dummyNonCachingStream() {
			yield { type: "content_block_start", content_block: { type: "text", text: "response" }, index: 0 }
			yield { type: "message_start", message: { usage: { input_tokens: 123, output_tokens: 456 } } }
		}

		it("should apply ephemeral caching to system prompt and last two user messages for caching models", async () => {
			// Create a fake function to simulate promptCaching.messages.create
			let createCallArgs: any
			let createCallHeaders: any
			const fakeCreate = async function* (args: any, headers: any) {
				createCallArgs = args
				createCallHeaders = headers
				yield* dummyStream()
			}
			// Fake client to simulate the beta branch for caching
			const fakeClient = {
				messages: {
					create: async function* () {
						throw new Error("Should not call non-caching create for caching model")
					},
				},
				beta: {
					promptCaching: {
						messages: {
							create: fakeCreate,
						},
					},
				},
			}
			// Options specifying a caching model id
			const options = {
				apiKey: "dummy",
				apiModelId: "claude-3-5-sonnet-20241022",
				anthropicBaseUrl: undefined,
			}
			const handler = new AnthropicHandler(options)
			// Override the client by casting to any
			;(handler as any).client = fakeClient

			const systemPrompt = "System message"
			const messages: Anthropic.Messages.MessageParam[] = [
				{ role: "user", content: "first" },
				{ role: "assistant", content: "assistant answer" },
				{ role: "user", content: "second" },
			]

			// Execute createMessage asynchronously to drain the stream
			const outputs: any[] = []
			for await (const output of handler.createMessage(systemPrompt, messages)) {
				outputs.push(output)
			}

			// System prompt must be set with cache_control as a cache breakpoint
			createCallArgs.system.should.deepEqual([
				{
					text: systemPrompt,
					type: "text",
					cache_control: { type: "ephemeral" },
				},
			])

			// Verify message transformations
			// First user message should be wrapped with cache_control
			createCallArgs.messages[0].should.deepEqual({
				role: "user",
				content: [
					{
						type: "text",
						text: "first",
						cache_control: { type: "ephemeral" },
					},
				],
			})

			// Assistant message should remain unchanged
			createCallArgs.messages[1].should.deepEqual({ role: "assistant", content: "assistant answer" })

			// Last user message should also be wrapped with cache_control
			createCallArgs.messages[2].should.deepEqual({
				role: "user",
				content: [
					{
						type: "text",
						text: "second",
						cache_control: { type: "ephemeral" },
					},
				],
			})

			// Verify headers for prompt caching
			createCallHeaders.should.deepEqual({
				headers: {
					"anthropic-beta": "prompt-caching-2024-07-31",
				},
			})
		})

		it("should not modify messages or system prompt for non-caching models", async () => {
			// Create a fake function to simulate client.messages.create
			let createCallArgs: any
			const fakeCreate = async function* (args: any) {
				createCallArgs = args
				yield* dummyNonCachingStream()
			}
			// Fake client mimicking the default branch

			const fakeClient = {
				messages: {
					create: fakeCreate,
				},
				beta: {
					promptCaching: {
						messages: {
							create: async function* () {
								throw new Error("Should not call caching create for non-caching model")
							},
						},
					},
				},
			}
			// Options specifying a non-caching model id
			const options = {
				apiKey: "dummy",
				apiModelId: "non-caching-model",
				anthropicBaseUrl: undefined,
			}
			const handler = new AnthropicHandler(options)
			// Override the client by casting to any
			;(handler as any).client = fakeClient

			const systemPrompt = "System message"
			const messages: Anthropic.Messages.MessageParam[] = [{ role: "user", content: "hello" }]

			const outputs: any[] = []
			for await (const output of handler.createMessage(systemPrompt, messages)) {
				outputs.push(output)
			}

			// The system prompt should be sent without cache_control
			createCallArgs.system.should.deepEqual([{ text: systemPrompt, type: "text" }])
			// The messages should remain unchanged
			createCallArgs.messages.should.deepEqual(messages)
		})

		it("should handle array content in messages for caching models", async () => {
			let createCallArgs: any
			const fakeCreate = async function* (args: any) {
				createCallArgs = args
				yield* dummyStream()
			}
			const fakeClient = {
				messages: {
					create: async function* () {
						throw new Error("Should not call non-caching create for caching model")
					},
				},
				beta: {
					promptCaching: {
						messages: {
							create: fakeCreate,
						},
					},
				},
			}
			const options = {
				apiKey: "dummy",
				apiModelId: "claude-3-5-sonnet-20241022",
			}
			const handler = new AnthropicHandler(options)
			;(handler as any).client = fakeClient

			const systemPrompt = "System message"
			const messages: Anthropic.Messages.MessageParam[] = [
				{
					role: "user",
					content: [
						{ type: "text" as const, text: "part 1" },
						{ type: "text" as const, text: "part 2" },
					],
				},
			]

			for await (const _ of handler.createMessage(systemPrompt, messages)) {
				// Drain the stream
			}

			// Only the last content block should have cache_control
			createCallArgs.messages[0].content[0].should.not.have.property("cache_control")
			createCallArgs.messages[0].content[1].should.have.property("cache_control")
			createCallArgs.messages[0].content[1].cache_control.should.deepEqual({ type: "ephemeral" })
		})
	})
})
