# Excerpt from src/online2/builder.py


# --- _modality (lines 149-154) ---
def _modality(path: Path) -> str:
    parts = set(path.parts)
    for name in ("E_environment", "A_actuator", "G_growth", "R_rootzone", "I_images"):
        if name in parts:
            return name
    return "interpretation" if "reference_answers" in parts else "metadata"

# --- IMAGE/INTERPRETATION event emission + embedding_status=pending (lines 350-410) ---
                point["events"][modality] = event_record
                if row_number % self.batch_size == 0:
                    self.db.commit()
        self.db.commit()

    def process_external(self) -> None:
        for path in self.files:
            modality = _modality(path)
            if modality not in {"I_images", "interpretation"}:
                continue
            relative = path.relative_to(self.root).as_posix()
            farm = path.parent.name if modality == "I_images" else path.stem.split("_")[0]
            set_name = self.case_sets.get(farm)
            decision = classify_public_path(path, set_name)
            if not decision.allowed or (
                modality == "interpretation" and decision.answer_tier != "score90"
            ):
                continue
            case_file = self.root / (set_name or "example_set") / "case_list.csv"
            with case_file.open(encoding="utf-8-sig", newline="") as handle:
                case = next((row for row in csv.DictReader(handle) if row["farm_id"] == farm), None)
            if case is None:
                continue
            source_file_id = stable_id(
                "sourcefile", {"path": relative, "checksum": self.checksums[path]}
            )
            self.emit("source_files", {
                "source_file_id": source_file_id, "path": relative,
                "checksum": self.checksums[path], "size_bytes": path.stat().st_size,
                "modality": modality,
            })
            timestamp, _ = _timestamp(case["period_end"])
            precision = "period"
            event_kind = "IMAGE" if modality == "I_images" else "INTERPRETATION"
            event_id = stable_id("event", {
                "kind": event_kind, "source": relative, "checksum": self.checksums[path],
            })
            group_id = self._emit_group(timestamp, precision)
            token = "IMAGE_EMBED_SLOT" if modality == "I_images" else "TEXT_EMBED_SLOT"
            token_id = stable_id("token", {"token": token, "version": "1"})
            self.emit("events", {
                "segment_id": event_id, "event_id": event_id, "event_kind": event_kind,
                "observation_timestamp": timestamp, "annotation_timestamp": "",
                "timestamp_precision": precision, "time_rank": 0,
                "same_time_group_id": group_id, "event_view": modality,
                "spatial_scope_type": "FARM", "spatial_scope_id": farm,
                "farm_ids": _json([farm]), "zone_ids": "[]", "source_set": set_name or "",
                "modalities": _json([modality]), "quality_flags": _json(["PERIOD_ALIGNED"]),
                "embedding_status": "pending",
            })
            self.emit("event_tokens", {
                "event_id": event_id, "token_id": token_id, "token_string": token,
                "position": 0, "role": "embedding_slot", "cell_id": "",
            })
            self.emit("external_embeddings", {
                "event_id": event_id, "modality": modality, "source_ref": relative,
                "embedding_ref": "", "model_id": "", "embedding_status": "pending",
                "answer_tier": decision.answer_tier or "",
                "quality_score": decision.quality_score or 0,
            })
            self.event_index[(farm, modality)].append({
