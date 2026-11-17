import argparse
import asyncio
import os
from dotenv import load_dotenv
from src.strategy_agent import generate_offline_briefing, run_agentic_triage
from src.stream_simulator import replay_race_stream

load_dotenv(override=True)


def main():
    parser = argparse.ArgumentParser(description="F1 Real-Time Telemetry + Agentic MCP Triage")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--gp", type=str, default="Silverstone")
    parser.add_argument("--session", type=str, default="R")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--driver", type=str, default="LEC")
    parser.add_argument("--rival", type=str, default="VER")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Force offline mode even if OPENAI_API_KEY is set")
    args = parser.parse_args()

    driver, rival = args.driver.upper(), args.rival.upper()

    if args.stream:
        replay_race_stream(args.year, args.gp, args.session, driver, rival, delay_sec=0.02, dataset_id=args.dataset)
        return

    if not args.offline and os.getenv("OPENAI_API_KEY"):
        briefing = asyncio.run(run_agentic_triage(driver, rival, args.year, args.gp, args.session, dataset_id=args.dataset))
    else:
        briefing = generate_offline_briefing(driver, rival, args.year, args.gp, args.session, dataset_id=args.dataset)

    print(briefing)


if __name__ == "__main__":
    main()