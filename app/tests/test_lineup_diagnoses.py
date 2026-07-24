from analytics.lineup_synergy import generate_lineup_diagnoses


def player(
    player_id,
    name,
    *,
    shooting,
    scoring=55.0,
    playmaking=45.0,
    is_shooter=False,
    fg3m_per36=0.8,
):
    return {
        "id": player_id,
        "name": name,
        "shooting": shooting,
        "scoring": scoring,
        "playmaking": playmaking,
        "rebounding": 50.0,
        "rim_pressure": 50.0,
        "defense": 50.0,
        "is_shooter": is_shooter,
        "fg3m_per36": fg3m_per36,
    }


def test_spacing_diagnosis_does_not_call_capable_players_non_threats():
    descriptors = [
        player(1, "LeBron James", shooting=48.0, scoring=85.0, playmaking=92.0),
        player(2, "Joel Embiid", shooting=24.0, scoring=82.0),
        player(3, "Interior Big", shooting=18.0, scoring=50.0),
        player(4, "Combo Guard", shooting=61.0, is_shooter=True, fg3m_per36=2.0),
        player(5, "Two-Way Wing", shooting=58.0, is_shooter=True, fg3m_per36=1.9),
    ]
    dims = {
        "playmaking": 78.0,
        "spacing": 35.0,
        "rebounding": 50.0,
        "paint": 55.0,
        "defense": 50.0,
        "scoring": 65.0,
        "pace": 50.0,
    }
    composition = {
        "creators": 2,
        "shooters": 2,
        "playmakers": 1,
        "bigs": 2,
    }
    channels = {
        "playmaking": 0.5,
        "spacing": -1.2,
        "interior": 0.0,
        "defense": 0.0,
        "overlap": 0.0,
    }

    strengths, weaknesses, *_ = generate_lineup_diagnoses(
        descriptors=descriptors,
        dims=dims,
        composition=composition,
        synergy_score=50.0,
        channels=channels,
    )

    cramped = next(item for item in weaknesses if item.startswith("Cramped Driving Lanes"))
    assert "The spacing around LeBron James is tight." in cramped
    assert "lower three-point volume from Interior Big and Joel Embiid" in cramped
    assert "LeBron James and" not in cramped
    assert "no perimeter threat" not in cramped
    assert "—" not in " ".join(strengths + weaknesses)
