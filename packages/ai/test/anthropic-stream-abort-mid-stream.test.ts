import { describe, expect, it } from "vitest";
import { stream } from "../src/api/anthropic-messages.ts";
import type { Context, Model } from "../src/types.ts";

function testModel(): Model<"anthropic-messages"> {
	return {
		id: "claude-test",
		name: "Claude Test",
		api: "anthropic-messages",
		provider: "anthropic",
		baseUrl: "http://127.0.0.1:9",
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 200000,
		maxTokens: 32000,
	};
}

const SSE_PREFIX = [
	`event: message_start\ndata: ${JSON.stringify({
		type: "message_start",
		message: {
			id: "msg_1",
			type: "message",
			role: "assistant",
			model: "claude-test",
			content: [],
			stop_reason: null,
			usage: { input_tokens: 1, output_tokens: 1 },
		},
	})}\n\n`,
	`event: content_block_start\ndata: ${JSON.stringify({
		type: "content_block_start",
		index: 0,
		content_block: { type: "text", text: "" },
	})}\n\n`,
	`event: content_block_delta\ndata: ${JSON.stringify({
		type: "content_block_delta",
		index: 0,
		delta: { type: "text_delta", text: "hello" },
	})}\n\n`,
].join("");

/**
 * A response that streams a little and then goes silent forever without closing
 * -- the shape of a provider still holding the connection open mid-response.
 *
 * The second `reader.read()` therefore never settles on its own. Polling
 * `signal.aborted` between reads cannot rescue that, because the check is only
 * reached if a read returns. The loop must react to the signal *while* a read is
 * pending, or the generator never returns and the whole turn hangs.
 */
function stallingFetch(): typeof fetch {
	return (async () => {
		const body = new ReadableStream<Uint8Array>({
			start(controller) {
				controller.enqueue(new TextEncoder().encode(SSE_PREFIX));
			},
			pull() {
				// Never resolves: no more data, no close, no error.
				return new Promise<void>(() => {});
			},
		});
		return new Response(body, {
			status: 200,
			headers: { "content-type": "text/event-stream" },
		});
	}) as unknown as typeof fetch;
}

describe("Anthropic stream abort mid-stream", () => {
	it("ends the turn when the signal fires while a read is pending", async () => {
		const context: Context = {
			messages: [{ role: "user", content: "hi", timestamp: Date.now() }],
		};
		const controller = new AbortController();

		const response = stream(testModel(), context, {
			apiKey: "test-key",
			cacheRetention: "none",
			signal: controller.signal,
			fetch: stallingFetch(),
		});

		const drained = (async () => {
			for await (const event of response) {
				// Abort as soon as the model is visibly streaming, then stop reading.
				if (event.type === "text_delta") {
					controller.abort();
					return "streamed-then-aborted";
				}
			}
			return "ended-without-delta";
		})();

		// Before the fix this never settles, so the assertion below is what a hang
		// looks like as a test failure rather than a wedged runner.
		const outcome = await Promise.race([
			drained,
			new Promise<string>((resolve) => setTimeout(() => resolve("HUNG"), 5000)),
		]);
		expect(outcome).toBe("streamed-then-aborted");

		const message = await Promise.race([
			response.result(),
			new Promise<never>((_resolve, reject) =>
				setTimeout(() => reject(new Error("result() never settled after abort")), 5000),
			),
		]);
		expect(message.stopReason).toBe("aborted");
	}, 20000);
});
