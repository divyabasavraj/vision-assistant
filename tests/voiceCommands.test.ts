import { describe, it, expect } from "vitest";
import { parseCommand } from "../lib/voiceCommands";

describe("parseCommand", () => {
  it("ignores text without the trigger word", () => {
    expect(parseCommand("what is this")).toBeNull();
  });
  it("extracts the question after the trigger", () => {
    expect(parseCommand("Vision, what's this?")).toEqual({ kind: "ask", question: "what's this" });
  });
  it("handles the trigger in the middle of noise", () => {
    expect(parseCommand("um okay vision read the sign please")).toEqual({ kind: "ask", question: "read the sign please" });
  });
  it("maps pause/resume/describe/read to local commands", () => {
    expect(parseCommand("vision pause")).toEqual({ kind: "pause" });
    expect(parseCommand("vision, stop")).toEqual({ kind: "pause" });
    expect(parseCommand("vision resume")).toEqual({ kind: "resume" });
    expect(parseCommand("vision describe")).toEqual({ kind: "describe" });
    expect(parseCommand("vision read")).toEqual({ kind: "read" });
  });
  it("returns empty when nothing follows the trigger", () => {
    expect(parseCommand("vision")).toEqual({ kind: "empty" });
    expect(parseCommand("vision.")).toEqual({ kind: "empty" });
  });
  it("supports a custom trigger word", () => {
    expect(parseCommand("hey buddy what is that", "buddy")).toEqual({ kind: "ask", question: "what is that" });
  });
});
