SHELL:=/bin/bash

BASH_COMPLETION_DIR?=/usr/share/bash-completion.d
BIN_DIR?=/usr/bin
DOC_DIR?=/usr/share/doc
MAN_DIR?=/usr/share/man
SHARE_DIR?=/usr/share
DEST_DIR?=


ifdef VERBOSE
  Q :=
else
  Q := @
endif


clean:
	$(Q)rm -rf ./build ./dist
	$(Q)find . -name __pycache__ -exec rm -rf {} \;


test:
	$(Q)python3 -m unittest


changelog.latest.md:
	$(Q)( \
		declare TAGS=(`git tag`); \
		for ((i=$${#TAGS[@]};i>=0;i--)); do \
			if [ $$i -eq 0 ]; then \
				echo -e "$${TAGS[$$i]}" >> changelog.latest.md; \
				git log $${TAGS[$$i]} --no-merges --format="  * %h %s"  >> changelog.latest.md; \
			elif [ $$i -eq $${#TAGS[@]} ] && [ $$(git log $${TAGS[$$i-1]}..HEAD --oneline | wc -l) -ne 0 ]; then \
				echo -e "$${TAGS[$$i-1]}-$$(git log -n 1 --format='%h')" >> changelog.latest.md; \
				git log $${TAGS[$$i-1]}..HEAD --no-merges --format="  * %h %s"  >> changelog.latest.md; \
			elif [ $$i -lt $${#TAGS[@]} ]; then \
				echo -e "$${TAGS[$$i]}" >> changelog.latest.md; \
				git log $${TAGS[$$i-1]}..$${TAGS[$$i]} --no-merges --format="  * %h %s"  >> changelog.latest.md; \
				break; \
			fi; \
		done \
	)
