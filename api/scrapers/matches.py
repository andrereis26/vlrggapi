import asyncio
import logging
import re
from collections import Counter
from datetime import datetime, timedelta

from utils.cache_manager import cache_manager
from utils.constants import (
    CACHE_TTL_LIVE,
    CACHE_TTL_RESULTS,
    CACHE_TTL_UPCOMING,
    LIVE_DETAIL_FETCH_CONCURRENCY,
    LIVE_DETAIL_FETCH_TIMEOUT,
    MATCH_DETAIL_TAB_FETCH_CONCURRENCY,
    MATCH_DETAIL_TAB_FETCH_TIMEOUT,
    VLR_BASE_URL,
    VLR_MATCHES_URL,
)
from utils.error_handling import handle_scraper_errors, raise_for_upstream_status
from utils.html_parsers import (
    HTMLParser,
    build_full_url,
    extract_match_teams,
    extract_text_content,
    normalize_image_url,
    parse_match_datetime_local,
    parse_href_id_slug,
    parse_html,
    parse_match_timestamp,
    parse_vlr_data_timestamp,
)
from utils.http_client import fetch_with_retries, get_http_client
from utils.pagination import PaginationConfig, scrape_multiple_pages

logger = logging.getLogger(__name__)


def _safe_flag(team_node) -> str:
    """Safely extract the homepage flag token from a team node."""
    flag_elem = team_node.css_first(".flag") if team_node else None
    if not flag_elem:
        return ""
    flag_class = flag_elem.attributes.get("class", "")
    return flag_class.replace(" mod-", "").replace("16", "_")


def _match_key_from_href(href: str) -> str:
    match_id, _ = parse_href_id_slug(href)
    return match_id or href.strip("/")


def _parse_api_timestamp(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None


def _iter_upcoming_page_items(html: HTMLParser):
    date_labels = html.css(".wf-label.mod-large")

    if not date_labels:
        for item in html.css("a.wf-module-item"):
            yield "", item
        return

    for label in date_labels:
        date_str = label.text().strip()
        sibling = label.next
        card = None
        while sibling is not None:
            if hasattr(sibling, 'tag') and sibling.tag and sibling.attributes:
                classes = sibling.attributes.get("class", "")
                if "wf-card" in classes:
                    card = sibling
                    break
            sibling = sibling.next
        if card is None:
            continue
        for item in card.css("a.wf-module-item"):
            yield date_str, item


def _infer_page_utc_offset(
    html: HTMLParser,
    reference_timestamps: dict[str, str] | None,
) -> timedelta | None:
    if not reference_timestamps:
        return None

    offsets: Counter[int] = Counter()
    for date_str, item in _iter_upcoming_page_items(html):
        href = item.attributes.get("href", "")
        reference = reference_timestamps.get(_match_key_from_href(href))
        if not reference:
            continue

        reference_dt = _parse_api_timestamp(reference)
        time_elem = item.css_first(".match-item-time")
        if reference_dt is None or time_elem is None:
            continue

        local_dt = parse_match_datetime_local(date_str, time_elem.text().strip())
        if local_dt is None:
            continue

        offset_seconds = round((local_dt - reference_dt).total_seconds() / 60) * 60
        if -12 * 3600 <= offset_seconds <= 14 * 3600:
            offsets[int(offset_seconds)] += 1

    if not offsets:
        return None

    return timedelta(seconds=offsets.most_common(1)[0][0])


def _homepage_upcoming_timestamps(html: HTMLParser) -> dict[str, str]:
    timestamps: dict[str, str] = {}

    for item in html.css(".js-home-matches-upcoming a.wf-module-item"):
        href = item.attributes.get("href", "")
        match_key = _match_key_from_href(href)
        if not match_key:
            continue

        timestamp = parse_match_timestamp(item, "", allow_eta=False)
        if timestamp:
            timestamps[match_key] = timestamp

    return timestamps


async def _fetch_homepage_upcoming_timestamps(
    *,
    max_retries: int,
    request_delay: float,
    timeout: int,
) -> dict[str, str]:
    try:
        client = get_http_client()
        resp = await fetch_with_retries(
            VLR_BASE_URL,
            client=client,
            timeout=timeout,
            max_retries=max_retries,
            request_delay=request_delay,
        )
        if resp.status_code != 200:
            logger.warning("Homepage timestamp reference returned status %d", resp.status_code)
            return {}

        return _homepage_upcoming_timestamps(parse_html(resp.text))
    except Exception as e:
        logger.warning("Failed to fetch homepage timestamp reference: %s", e)
        return {}


def _extract_detail_timestamp(html: HTMLParser) -> str:
    ts_elem = html.css_first(".match-header-date .moment-tz-convert")
    if ts_elem is None:
        return ""

    return parse_vlr_data_timestamp(ts_elem.attributes.get("data-utc-ts", ""))


async def _fill_missing_upcoming_timestamps(
    data: dict,
    *,
    max_retries: int,
    request_delay: float,
) -> None:
    segments = data.get("data", {}).get("segments", [])
    missing_segments = [
        segment
        for segment in segments
        if not segment.get("unix_timestamp") and segment.get("match_page")
    ]
    if not missing_segments:
        return

    client = get_http_client()
    semaphore = asyncio.Semaphore(MATCH_DETAIL_TAB_FETCH_CONCURRENCY)

    async def fetch_timestamp(segment):
        try:
            async with semaphore:
                resp = await fetch_with_retries(
                    segment["match_page"],
                    client=client,
                    timeout=MATCH_DETAIL_TAB_FETCH_TIMEOUT,
                    max_retries=max_retries,
                    request_delay=request_delay,
                )
            if resp.status_code != 200:
                return

            timestamp = _extract_detail_timestamp(parse_html(resp.text))
            if timestamp:
                segment["unix_timestamp"] = timestamp
        except Exception as e:
            logger.warning("Failed to fetch detail timestamp %s: %s", segment["match_page"], e)

    await asyncio.gather(*(fetch_timestamp(segment) for segment in missing_segments))




@handle_scraper_errors
async def vlr_upcoming_matches(num_pages=1, from_page=None, to_page=None):
    """Get upcoming matches from VLR.GG homepage."""
    async def build():
        client = get_http_client()
        resp = await fetch_with_retries(VLR_BASE_URL, client=client)
        status = resp.status_code
        raise_for_upstream_status(status, "upcoming matches")

        html = parse_html(resp.text)

        result = []
        for item in html.css(".js-home-matches-upcoming a.wf-module-item"):
            is_upcoming = item.css_first(".h-match-eta.mod-upcoming")
            if not is_upcoming:
                continue

            team1, team2 = extract_match_teams(item, ".h-match-team")

            eta = extract_text_content(item.css_first(".h-match-eta"))
            if eta != "LIVE":
                eta = eta + " from now"

            match_event = extract_text_content(item.css_first(".h-match-preview-event"))
            match_series = extract_text_content(item.css_first(".h-match-preview-series"))
            timestamp = parse_match_timestamp(item, "")
            url_path = build_full_url(item.attributes.get("href", ""))

            result.append(
                {
                    "team1": team1["name"],
                    "team2": team2["name"],
                    "flag1": team1["flag"],
                    "flag2": team2["flag"],
                    "time_until_match": eta,
                    "match_series": match_series,
                    "match_event": match_event,
                    "unix_timestamp": timestamp,
                    "match_page": url_path,
                }
            )

        data = {"data": {"status": status, "segments": result}}

        return data

    return await cache_manager.get_or_create_async(CACHE_TTL_UPCOMING, build, "upcoming")


@handle_scraper_errors
async def vlr_live_score(num_pages=1, from_page=None, to_page=None):
    """Get live match scores from VLR.GG. Fetches match detail pages concurrently."""
    async def build():
        client = get_http_client()
        resp = await fetch_with_retries(VLR_BASE_URL, client=client)
        status = resp.status_code
        raise_for_upstream_status(status, "live scores")

        html = parse_html(resp.text)

        matches = html.css(".js-home-matches-upcoming a.wf-module-item")
        live_matches = []
        for match in matches:
            is_live = match.css_first(".h-match-eta.mod-live")
            if not is_live:
                continue

            teams = []
            flags = []
            scores = []
            round_texts = []
            for team in match.css(".h-match-team"):
                teams.append(extract_text_content(team.css_first(".h-match-team-name")) or "TBD")
                flags.append(_safe_flag(team))
                scores.append(extract_text_content(team.css_first(".h-match-team-score")))
                round_info_ct = team.css(".h-match-team-rounds .mod-ct")
                round_info_t = team.css(".h-match-team-rounds .mod-t")
                round_text_ct = round_info_ct[0].text().strip() if round_info_ct else "N/A"
                round_text_t = round_info_t[0].text().strip() if round_info_t else "N/A"
                round_texts.append({"ct": round_text_ct, "t": round_text_t})

            while len(teams) < 2:
                teams.append("TBD")
            while len(flags) < 2:
                flags.append("")
            while len(scores) < 2:
                scores.append("")
            while len(round_texts) < 2:
                round_texts.append({"ct": "N/A", "t": "N/A"})

            match_event = extract_text_content(match.css_first(".h-match-preview-event"))
            match_series = extract_text_content(match.css_first(".h-match-preview-series"))
            timestamp = parse_match_timestamp(match, "")
            href = match.attributes.get("href", "")
            url_path = build_full_url(href)
            match_id, _ = parse_href_id_slug(href)

            live_matches.append({
                "teams": teams,
                "flags": flags,
                "scores": scores,
                "round_texts": round_texts,
                "match_event": match_event,
                "match_series": match_series,
                "timestamp": timestamp,
                "url_path": url_path,
                "match_id": match_id,
            })

        detail_fetch_semaphore = asyncio.Semaphore(LIVE_DETAIL_FETCH_CONCURRENCY)

        async def fetch_match_detail(url):
            try:
                async with detail_fetch_semaphore:
                    return await fetch_with_retries(
                        url,
                        client=client,
                        timeout=LIVE_DETAIL_FETCH_TIMEOUT,
                        max_retries=1,
                    )
            except Exception as e:
                logger.warning("Failed to fetch match detail %s: %s", url, e)
                return None

        detail_responses = await asyncio.gather(
            *[fetch_match_detail(m["url_path"]) for m in live_matches]
        )

        result = []
        for match_data, detail_resp in zip(live_matches, detail_responses):
            team_logos = ["", ""]
            current_map = "Unknown"
            map_number = "Unknown"

            if detail_resp is not None:
                match_html = parse_html(detail_resp.text)

                logos = []
                for img in match_html.css(".match-header-vs img"):
                    logo_url = "https:" + img.attributes.get("src", "")
                    logos.append(logo_url)
                if len(logos) >= 2:
                    team_logos = logos[:2]

                current_map_element = match_html.css_first(
                    ".vm-stats-gamesnav-item.js-map-switch.mod-active.mod-live"
                )
                if current_map_element:
                    map_text = (
                        current_map_element.css_first("div", default="Unknown")
                        .text().strip().replace("\n", "").replace("\t", "")
                    )
                    current_map = re.sub(r"^\d+", "", map_text)
                    map_number_match = re.search(r"^\d+", map_text)
                    map_number = map_number_match.group(0) if map_number_match else "Unknown"

            rt = match_data["round_texts"]
            result.append(
                {
                    "team1": match_data["teams"][0],
                    "team2": match_data["teams"][1],
                    "flag1": match_data["flags"][0],
                    "flag2": match_data["flags"][1],
                    "team1_logo": team_logos[0],
                    "team2_logo": team_logos[1],
                    "score1": match_data["scores"][0],
                    "score2": match_data["scores"][1],
                    "team1_round_ct": rt[0]["ct"] if len(rt) > 0 else "N/A",
                    "team1_round_t": rt[0]["t"] if len(rt) > 0 else "N/A",
                    "team2_round_ct": rt[1]["ct"] if len(rt) > 1 else "N/A",
                    "team2_round_t": rt[1]["t"] if len(rt) > 1 else "N/A",
                    "map_number": map_number,
                    "current_map": current_map,
                    "time_until_match": "LIVE",
                    "match_event": match_data["match_event"],
                    "match_series": match_data["match_series"],
                    "unix_timestamp": match_data["timestamp"],
                    "match_page": match_data["url_path"],
                    "match_id": match_data["match_id"],
                }
            )

        data = {"data": {"status": status, "segments": result}}

        return data

    return await cache_manager.get_or_create_async(CACHE_TTL_LIVE, build, "live_score")


def _parse_single_match(item, date_str, page, page_utc_offset: timedelta | None = None):
    """Extract all match fields from one <a> element. Returns dict or None."""
    eta_element = item.css_first(".ml-eta")
    if eta_element and "ago" in eta_element.text():
        return None

    href = item.attributes.get("href", "")
    url_path = "https://www.vlr.gg" + href if href else ""

    eta = item.css_first(".ml-status").text().strip() if item.css_first(".ml-status") else ""
    if not eta:
        eta_elem = item.css_first(".ml-eta")
        if eta_elem:
            eta_text = eta_elem.text().strip()
            if eta_text and "ago" not in eta_text:
                eta = eta_text

    teams = []
    flags = []
    scores_list = []
    for team_div in item.css(".match-item-vs-team"):
        team_name_elem = team_div.css_first(".match-item-vs-team-name")
        teams.append(team_name_elem.text().strip() if team_name_elem else "TBD")

        flag_elem = team_div.css_first(".flag")
        if flag_elem:
            flag_class = flag_elem.attributes.get("class")
            flags.append(flag_class.replace("flag ", "").replace(" mod-", "_") if flag_class else "")
        else:
            flags.append("")

        score_elem = team_div.css_first(".match-item-vs-team-score")
        scores_list.append(score_elem.text().strip() if score_elem else "")

    while len(teams) < 2:
        teams.append("TBD")
    while len(flags) < 2:
        flags.append("")
    while len(scores_list) < 2:
        scores_list.append("")

    match_event_elem = item.css_first(".match-item-event-series")
    match_series = ""
    if match_event_elem:
        event_text = match_event_elem.text().replace("\n", "").replace("\t", "").strip()
        parts = event_text.split()
        if parts:
            match_series = " ".join(parts)

    tourney_elem = item.css_first(".match-item-event")
    tourney = ""
    if tourney_elem:
        tourney_lines = [line.strip() for line in tourney_elem.text().split("\n") if line.strip()]
        tourney = tourney_lines[-1] if tourney_lines else ""

    tourney_icon_elem = item.css_first(".match-item-icon img")
    tourney_icon_url = ""
    if tourney_icon_elem:
        icon_src = tourney_icon_elem.attributes.get("src", "")
        if icon_src:
            tourney_icon_url = normalize_image_url(icon_src)

    timestamp = parse_match_timestamp(
        item,
        date_str,
        page_utc_offset=page_utc_offset,
        prefer_date_time=bool(date_str),
        allow_eta=not bool(date_str),
    )

    return {
        "team1": teams[0],
        "team2": teams[1],
        "flag1": flags[0],
        "flag2": flags[1],
        "score1": scores_list[0],
        "score2": scores_list[1],
        "time_until_match": eta,
        "match_series": match_series,
        "match_event": tourney,
        "unix_timestamp": timestamp,
        "match_page": url_path,
        "tournament_icon": tourney_icon_url,
        "page_number": page,
    }


def _parse_upcoming_page(
    html: HTMLParser,
    page: int,
    reference_timestamps: dict[str, str] | None = None,
    page_utc_offset: timedelta | None = None,
) -> list[dict]:
    """Parse callback for scrape_multiple_pages — upcoming extended matches."""
    page_results = []
    page_utc_offset = (
        _infer_page_utc_offset(html, reference_timestamps)
        or page_utc_offset
    )

    for date_str, item in _iter_upcoming_page_items(html):
        try:
            match_data = _parse_single_match(item, date_str, page, page_utc_offset)
            if match_data is not None:
                page_results.append(match_data)
        except Exception as e:
            logger.warning("Failed to parse match on page %d: %s", page, e)

    return page_results


def _parse_results_page(html: HTMLParser, page: int) -> list[dict]:
    """Parse callback for scrape_multiple_pages — match results."""
    page_results = []
    items = html.css("a.wf-module-item")

    for item in items:
        try:
            href = item.attributes["href"]
            url_path = build_full_url(href)
            eta = item.css_first("div.ml-eta").text() + " ago"
            rounds = (
                item.css_first("div.match-item-event-series")
                .text()
                .replace("\u2013", "-")
                .replace("\n", "")
                .replace("\t", "")
            )
            tourney = (
                item.css_first("div.match-item-event")
                .text()
                .replace("\t", " ")
                .strip()
                .split("\n")[1]
                .strip()
            )
            tourney_icon_url = f"https:{item.css_first('img').attributes['src']}"

            try:
                team_array = (
                    item.css_first("div.match-item-vs").css_first("div:nth-child(2)").text()
                )
            except Exception:
                team_array = "TBD"
            team_array = (
                team_array.replace("\t", " ")
                .replace("\n", " ")
                .strip()
                .split("                                  ")
            )
            team1 = team_array[0]
            score1 = team_array[1].replace(" ", "").strip()
            team2 = team_array[4].strip()
            score2 = team_array[-1].replace(" ", "").strip()

            flag_list = [
                flag_parent.attributes["class"].replace(" mod-", "_")
                for flag_parent in item.css(".flag")
            ]
            flag1 = flag_list[0] if len(flag_list) > 0 else ""
            flag2 = flag_list[1] if len(flag_list) > 1 else ""

            page_results.append(
                {
                    "team1": team1,
                    "team2": team2,
                    "score1": score1,
                    "score2": score2,
                    "flag1": flag1,
                    "flag2": flag2,
                    "time_completed": eta,
                    "round_info": rounds,
                    "tournament_name": tourney,
                    "match_page": url_path,
                    "tournament_icon": tourney_icon_url,
                    "page_number": page,
                }
            )
        except Exception as e:
            logger.warning("Failed to parse result on page %d: %s", page, e)
            continue

    return page_results


@handle_scraper_errors
async def vlr_upcoming_matches_extended(
    num_pages=1, from_page=None, to_page=None,
    max_retries=3, request_delay=1.0, timeout=30,
):
    """Scrape upcoming matches from the paginated matches page."""
    config = PaginationConfig(
        num_pages=num_pages, from_page=from_page, to_page=to_page,
        max_retries=max_retries, request_delay=request_delay, timeout=timeout,
    )
    cache_key = ("upcoming_ext", num_pages, from_page, to_page)

    async def build():
        reference_timestamps = await _fetch_homepage_upcoming_timestamps(
            max_retries=max_retries,
            request_delay=request_delay,
            timeout=timeout,
        )
        last_page_utc_offset: timedelta | None = None

        def parse_upcoming_page(html: HTMLParser, page: int) -> list[dict]:
            nonlocal last_page_utc_offset
            page_utc_offset = _infer_page_utc_offset(html, reference_timestamps)
            if page_utc_offset is not None:
                last_page_utc_offset = page_utc_offset
            else:
                page_utc_offset = last_page_utc_offset

            return _parse_upcoming_page(
                html,
                page,
                page_utc_offset=page_utc_offset,
            )

        data = await scrape_multiple_pages(
            base_url=VLR_MATCHES_URL,
            parse_func=parse_upcoming_page,
            config=config,
        )
        await _fill_missing_upcoming_timestamps(
            data,
            max_retries=max_retries,
            request_delay=request_delay,
        )
        return data

    return await cache_manager.get_or_create_async(CACHE_TTL_UPCOMING, build, *cache_key)


@handle_scraper_errors
async def vlr_match_results(
    num_pages=1, from_page=None, to_page=None,
    max_retries=3, request_delay=1.0, timeout=30,
):
    """Scrape match results with pagination."""
    config = PaginationConfig(
        num_pages=num_pages, from_page=from_page, to_page=to_page,
        max_retries=max_retries, request_delay=request_delay, timeout=timeout,
    )
    cache_key = ("results", num_pages, from_page, to_page)

    async def build():
        return await scrape_multiple_pages(
            base_url=f"{VLR_MATCHES_URL}/results",
            parse_func=_parse_results_page,
            config=config,
        )

    return await cache_manager.get_or_create_async(CACHE_TTL_RESULTS, build, *cache_key)
