# =============================================================================
# @file    Makefile
# @brief   Makefile for generating previews of the paper
# @author  Michael Hucka <mhucka@caltech.edu>
# @license Please see the file named LICENSE in the project directory
# @website https://github.com/casics/dassie
# =============================================================================

# Change the following values to match your configuration.
# .............................................................................

input	:= paper.md
output	:= paper.pdf
bib	:= paper.bib

# Main code -- no more customization variables after this point
# .............................................................................

title	:= $(shell grep title: $(input) | sed 's/title: *//' | tr -d "'")
authors := $(shell sed -n '/authors:/,/affiliations:/p' $(input) | grep name: | sed 's/- name: *//' | paste -d, -s - | sed 's/,/, /g')
repo	:= $(shell git remote get-url origin | sed 's|git@github.com:|https://github.com/|' | sed 's/\.git//')

$(output): $(input) $(bib) Makefile
	/usr/bin/pandoc \
	-V paper_title="$(title)" \
	-V citation_author="$(authors)" \
	-V repository="$(repo)" \
	-V archive_doi="http://dx.doi.org/00.00000/zenodo.0000000" \
	-V formatted_doi="00.00000/joss.00000" \
	-V paper_url="http://joss.theoj.org/papers/" \
	-V review_issue_url="http://joss.theoj.org/papers/" \
	-V issue="0" \
	-V volume="00" \
	-V year="2018" \
	-V submitted="00 January 0000" \
	-V published="00 January 0000" \
	-V page="00" \
	-V graphics="true" \
	-V geometry:margin=1in \
	-V logo_path:"joss-logo.png" \
	-o $(output) \
	--pdf-engine=xelatex \
	--filter pandoc-citeproc $(input) \
	--from markdown+autolink_bare_uris \
	--template "latex.template"

autorefresh:;
	((ls $(input) $(bib) | entr make $(output)) &)
