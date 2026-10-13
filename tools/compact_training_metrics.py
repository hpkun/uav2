"""Streaming offline JSONL compaction; originals are preserved unless --replace."""
from pathlib import Path
import argparse
import json
import os
import tempfile
import uuid


def file_identity(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino


def compact_metrics(input_path, output_path=None, interval=20000, replace=False):
    source = Path(input_path).resolve()
    if type(interval) is not int or interval <= 0:
        raise ValueError("interval-sampled-steps must be a positive integer")
    if replace and output_path is not None:
        raise ValueError("--replace cannot be combined with --output")
    before = file_identity(source)
    destination = Path(output_path).resolve() if output_path else source.with_name(f"{source.stem}.compact{source.suffix}")
    if not replace and (destination == source or destination.exists()):
        raise ValueError("output must be a new file distinct from input")
    input_rows = output_rows = 0
    first = last = last_written = next_threshold = None
    backup = None
    temporary = None
    created_output = False
    try:
        if replace:
            writer = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                dir=source.parent, prefix=f".{source.name}.", suffix=".tmp", delete=False)
            temporary = Path(writer.name)
        else:
            writer = destination.open("x",encoding="utf-8",newline="\n")
            created_output = True
        with writer, source.open(encoding="utf-8") as reader:
            def write(record):
                nonlocal output_rows,last_written
                writer.write(json.dumps(record,separators=(",", ":"))+"\n")
                output_rows += 1
                last_written = record["sampled_steps"]
            for number,line in enumerate(reader,1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except ValueError as error:
                    raise ValueError(f"invalid JSON at line {number}") from error
                steps = record.get("sampled_steps") if isinstance(record,dict) else None
                if type(steps) is not int or steps < 0:
                    raise ValueError(f"sampled_steps must be a nonnegative integer at line {number}")
                if last is not None and steps <= last["sampled_steps"]:
                    raise ValueError(f"sampled_steps must be strictly increasing at line {number}")
                input_rows += 1
                if first is None:
                    first = steps
                    write(record)
                    next_threshold = (steps // interval + 1)*interval
                elif steps >= next_threshold:
                    write(record)
                    next_threshold = (steps // interval + 1)*interval
                last = record
            if last is not None and last["sampled_steps"] != last_written:
                write(last)
            writer.flush()
            os.fsync(writer.fileno())
        output_bytes = (temporary if replace else destination).stat().st_size
        changed = file_identity(source) != before
        if replace:
            if changed or file_identity(source) != before:
                raise RuntimeError("input size/mtime/identity changed; refusing --replace on an active log")
            backup = source.with_name(f"{source.name}.backup.{uuid.uuid4().hex}")
            os.replace(source,backup)
            try:
                os.replace(temporary,source)
            except BaseException:
                os.replace(backup,source)
                raise
            temporary = None
        return {"input_rows":input_rows,"output_rows":output_rows,
            "input_bytes":before[0],"output_bytes":output_bytes,
            "compression_ratio":before[0]/output_bytes if output_bytes else None,
            "first_sampled_steps":first,"last_sampled_steps":last["sampled_steps"] if last else None,
            "input_changed_during_scan":changed,"output":str(source if replace else destination),
            "backup":str(backup) if backup else None}
    except BaseException:
        if temporary is not None and temporary.exists():
            temporary.unlink()
        if created_output:
            destination.unlink(missing_ok=True)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--interval-sampled-steps",type=int,default=20000)
    parser.add_argument("--replace",action="store_true")
    args=parser.parse_args()
    result=compact_metrics(args.input,args.output,args.interval_sampled_steps,args.replace)
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
