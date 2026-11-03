function Image(im)
 im.src = im.src:gsub("%.png$", ".pdf")
 return im
end
function Table(t)
 -- Set sensible widths for the wide, mostly numeric result tables.
 local n=#t.colspecs
 local first=pandoc.utils.stringify(t.head.rows[1].cells[1])
 local widths=nil
 if n==7 then widths={.19,.11,.11,.12,.13,.14,.20}
 elseif n==6 then widths={.18,.16,.16,.16,.17,.17}
 elseif n==3 and first=="Method" then widths={.19,.45,.36}
 elseif n==3 and first=="quantity" then widths={.26,.28,.46}
 elseif n==3 and first=="profile" then widths={.36,.32,.32}
 elseif n==4 and first:match("band") then widths={.19,.29,.30,.22}
 end
 if widths then for i,v in ipairs(widths) do t.colspecs[i]={t.colspecs[i][1],v} end end
 -- Keep small tables intact when they fit on one page.
 local rows = #t.head.rows
 for _, body in ipairs(t.bodies) do rows = rows + #body.head + #body.body end
 if rows <= 8 then
  local height = (rows + 3) * 12
  local guard = "\\KeepTableSpace{" .. height .. "pt}"
  return {pandoc.RawBlock("latex", guard), t}
 end
 return t
end

function Header(h)
 h.level = math.max(1, h.level - 1)
 if h.identifier == "introduction" then
  return {pandoc.RawBlock("latex", "\\clearpage"), h}
 end
 if h.identifier == "what-the-recorded-trajectories-show" then
  -- Give the section opening and its trajectory plot room together.
  return {pandoc.RawBlock("latex", "\\ifdim\\dimexpr\\pagegoal-\\pagetotal\\relax<0.7\\textheight\\clearpage\\fi"), h}
 elseif h.identifier == "references" then
  -- Keep the body type size and separate bibliography entries with a smaller gap.
  return {h, pandoc.RawBlock("latex", "\\setlength{\\parskip}{2pt plus 1pt}")}
 end
 return h
end
