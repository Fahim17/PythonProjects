
for i in range(300,382):
	n = str(i)
	fnm = "contacts/"+n+".vcf"
	# fnm = n+".vcf"
	f = open(fnm, "r")
	ot = open("cont.vcf", "a")
	try:		
		for x in f:
		  ot.write(str(x))
		print("Done: ",i)
	except:
		print("err", i)
		continue