import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import data from "./fixture.json";
import { App } from "../src/app/App";

const ok = async () => data;
afterEach(cleanup);

describe("dashboard", () => {
  it("shows a skeleton, then both phones and every section", async () => {
    render(<App fetcher={ok} />);
    expect(screen.getByRole("status")).toBeTruthy();
    expect(await screen.findByText("Against the truth")).toBeTruthy();
    expect(screen.getByText("Today")).toBeTruthy();
    expect(screen.getByText("Assembled")).toBeTruthy();
    expect(screen.getByText("Which A/B test to trust")).toBeTruthy();
  });

  it("highlights busy kitchens", async () => {
    render(<App fetcher={ok} />);
    await screen.findByText("Against the truth");
    fireEvent.click(screen.getByRole("radio", { name: "Busy kitchens" }));
    expect(screen.getAllByLabelText(/busy kitchen/).length).toBe(1);
  });

  it("switches the A/B section to the other treatment", async () => {
    render(<App fetcher={ok} />);
    await screen.findByText("Against the truth");
    fireEvent.click(screen.getByRole("radio", { name: "Re-ranked rows" }));
    expect(screen.getByText(/today's rows, re-ranked for the whole week/)).toBeTruthy();
  });

  it("offers a retry when the data does not load", async () => {
    let calls = 0;
    const flaky = async () => {
      calls += 1;
      if (calls === 1) throw new Error("data.json answered 404");
      return data;
    };
    render(<App fetcher={flaky} />);
    const button = await screen.findByRole("button", { name: "Load again" });
    await act(async () => fireEvent.click(button));
    expect(await screen.findByText("Against the truth")).toBeTruthy();
  });

  it("says what to run when there are no sample pages", async () => {
    const empty = async () => ({ ...data, samples: { ...data.samples, customers: [] } });
    render(<App fetcher={empty} />);
    expect(await screen.findByText(/No sample pages yet/)).toBeTruthy();
  });
});
