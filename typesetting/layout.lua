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
 return t
end

function Header(h)
 if h.identifier == "what-the-recorded-trajectories-show" then
  -- Give the section opening and its trajectory plot room together.
  return {pandoc.RawBlock("latex", "\\ifdim\\dimexpr\\pagegoal-\\pagetotal\\relax<0.7\\textheight\\clearpage\\fi"), h}
 elseif h.identifier == "references" then
  -- Keep the body type size and separate bibliography entries with a smaller gap.
  return {h, pandoc.RawBlock("latex", "\\setlength{\\parskip}{2pt plus 1pt}")}
 end
end
