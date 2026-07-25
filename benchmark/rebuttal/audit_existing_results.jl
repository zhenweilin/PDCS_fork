#!/usr/bin/env julia

using SHA
include("common.jl")
using .RebuttalCommon

root=abspath(option("root","rebuttal_plan"))
output=abspath(option("output",
 "benchmark/results/rebuttal/audit/summary_diff.csv"))
inventory=abspath(option("inventory",
 joinpath(dirname(output),"source_inventory.csv")))
files=sort(filter(p->endswith(p,".md") &&
                    occursin("result",lowercase(basename(p))),
                  collect(Iterators.flatten(
                    (joinpath(d,f) for f in fs) for (d,_,fs) in walkdir(root)))))

function cells(line)
    strip.(split(strip(line,[' ','|']),'|'))
end
function number(text)
    cleaned=replace(text,','=>"")
    found=match(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?",cleaned)
    found===nothing ? nothing : tryparse(Float64,found.match)
end
function canonical(text)
    lowercase(replace(strip(text),r"\s+"=>" ",r"[*`$]"=>""))
end

entries=NamedTuple[]
for path in files
    header=String[]
    for (line_number,line) in enumerate(eachline(path))
        startswith(strip(line),"|") || continue
        c=cells(line)
        all(x->occursin(r"^:?-+:?$",x),c) && continue
        if isempty(header)
            header=c
            continue
        end
        length(c)==length(header) || continue
        case_key=canonical(header[1])*"="*canonical(c[1])
        for j in 2:length(c)
            value=number(c[j]); value===nothing && continue
            push!(entries,(experiment=splitext(basename(path))[1],
              case=case_key,metric=canonical(header[j]),source=relpath(path,root),
              displayed=strip(c[j]),value=value,line=line_number))
        end
    end
end

groups=Dict{Tuple{String,String},Vector{eltype(entries)}}()
for e in entries
    push!(get!(groups,(e.case,e.metric),eltype(entries)[]),e)
end
rows=Any[]
for ((case_key,metric),group) in sort(collect(groups);by=first)
    length(group)>=2 || continue
    for i in 1:length(group)-1, j in i+1:length(group)
        a,b=group[i],group[j]
        status=isapprox(a.value,b.value;rtol=5e-4,atol=5e-12) ?
               "consistent" : "rerun_required"
        push!(rows,(a.experiment,case_key,metric,a.source,a.displayed,
                    b.source,b.displayed,status,a.line,b.line))
    end
end
write_csv(output,("experiment","case","metric","source_a","value_a",
 "source_b","value_b","status","line_a","line_b"),rows)
write_csv(inventory,("source","bytes","sha256"),
 [(relpath(p,root),filesize(p),bytes2hex(sha256(read(p)))) for p in files])
@info "Existing result audit complete" output comparisons=length(rows) sources=length(files)
