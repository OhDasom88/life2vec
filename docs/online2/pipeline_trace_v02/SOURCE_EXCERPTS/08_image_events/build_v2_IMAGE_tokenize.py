# Excerpt from scripts/online2_v2/build_v2.py tokenize_events


# --- IMAGE → [IMAGE_SLOT] sentence; embedding_status passthrough (lines 287-340) ---
        if ev.event_kind == "IMAGE":
            tokens = ["[IMAGE_SLOT]", "IMAGE_ROLE|UNRESOLVED", "VIEW|IMAGE", "EVENT_KIND|IMAGE"]
            group_ids = ["NONE"] * len(tokens)
            roles = ["slot", "meta", "meta", "meta"]
        elif ev.event_kind == "INTERPRETATION":
            tokens = ["[TEXT_SLOT]", "VIEW|INTERPRETATION", "EVENT_KIND|INTERPRETATION"]
            group_ids = ["NONE"] * len(tokens)
            roles = ["slot", "meta", "meta"]
        elif grp is not None:
            for cell in grp:
                column_name = cell[4]
                if column_name in skip_cols:
                    continue
                raw = "" if bool(cell[8]) else str(cell[7])
                measured = tokenizer.tokenize_value(
                    str(column_name),
                    raw,
                    farm_id=str(cell[6]),
                    cell_id=str(cell[5]),
                    atomic_value_id=str(cell[9] or ""),
                )
                tokens.append("[MEAS_SEP]")
                group_ids.append(measured.measurement_group_id)
                roles.append("sep")
                for tok, role in zip(measured.tokens, measured.roles):
                    tokens.append(tok)
                    group_ids.append(measured.measurement_group_id)
                    roles.append(role)
            view_name = str(ev.view)
            view_tok = view_name.split("_")[-1].upper() if "_" in view_name else view_name.upper()
            tokens = ["[EVENT_SEP]", f"VIEW|{view_tok}", "EVENT_KIND|OBSERVATION"] + tokens
            group_ids = ["NONE", "NONE", "NONE"] + group_ids
            roles = ["sep", "meta", "meta"] + roles
        else:
            tokens = ["[EVENT_SEP]", "EVENT_KIND|OBSERVATION", "[MISSING]"]
            group_ids = ["NONE"] * 3
            roles = ["sep", "meta", "quality"]

        token_ids = [vocab.get(t) for t in tokens]
        rows.append(
            {
                "event_id": ev.event_id,
                "same_time_group_id": ev.same_time_group_id,
                "event_kind": ev.event_kind,
                "observation_timestamp": ev.observation_timestamp,
                "farm_id": ev.farm,
                "zone_id": ev.zone,
                "SENTENCE": " ".join(tokens),
                "measurement_group_ids": json.dumps(group_ids),
                "token_roles": json.dumps(roles),
                "token_ids": json.dumps(token_ids),
                "embedding_status": getattr(ev, "embedding_status", ""),
                "tokenization_version": "v2",
            }
