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

const SSE_TAIL = [
	`event: content_block_stop\ndata: ${JSON.stringify({ type: "content_block_stop", index: 0 })}\n\n`,
	`event: message_delta\ndata: ${JSON.stringify({
		type: "message_delta",
		delta: { stop_reason: "end_turn", stop_sequence: null },
		usage: { output_tokens: 5 },
	})}\n\n`,
	`event: message_stop\ndata: ${JSON.stringify({ type: "message_stop" })}\n\n`,
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

/** A response that streams an entire turn and closes, the way every normal one does. */
function completingFetch(): typeof fetch {
	return (async () => {
		const body = new ReadableStream<Uint8Array>({
			start(controller) {
				controller.enqueue(new TextEncoder().encode(SSE_PREFIX + SSE_TAIL));
				controller.close();
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

	// Cancelling the reader is only half of it. A stream that ends normally
	// releases its reader in the `finally`, so a listener left on the caller's
	// signal outlives the reader it captured. A later abort -- the next Escape,
	// or a turn the user interrupts after this one -- then cancels a released
	// reader. That *rejects* rather than throwing, so it survives any try/catch
	// around the call, and unobserved it takes the process down.
	//
	// This is not hypothetical. A minified build carrying this fix without both
	// guards crashed real sessions on the second Escape with "TypeError: Invalid
	// state: The reader is not attached to a stream" (ERR_INVALID_STATE).
	//
	// Each guard alone is unobservable from here, which is exactly why they are
	// easy to drop: removing only the listener leaves the rejection handled, and
	// keeping only the listener cancels a reader that is still live. Counting
	// listeners on the signal does not separate them either, because the SDK
	// attaches one of its own per request and never removes it.
	it("leaves nothing behind that a later abort can trip over", async () => {
		const context: Context = {
			messages: [{ role: "user", content: "hi", timestamp: Date.now() }],
		};
		const controller = new AbortController();

		const response = stream(testModel(), context, {
			apiKey: "test-key",
			cacheRetention: "none",
			signal: controller.signal,
			fetch: completingFetch(),
		});
		for await (const _event of response) {
			// Drain the turn to completion; nothing is aborted here.
		}
		expect((await response.result()).stopReason).toBe("stop");

		const rejections: unknown[] = [];
		const capture = (reason: unknown) => rejections.push(reason);
		process.on("unhandledRejection", capture);
		try {
			// The user presses Escape after the turn has already ended.
			controller.abort();
			// unhandledRejection is emitted after the microtask queue drains.
			await new Promise((resolve) => setTimeout(resolve, 100));
		} finally {
			process.off("unhandledRejection", capture);
		}
		expect(rejections.map(String)).toEqual([]);
	}, 20000);
});
