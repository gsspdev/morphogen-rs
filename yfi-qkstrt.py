import yfinance as yf
dat = yf.Ticker("MSFT")

info = dat.info
calendar = dat.calendar
price_targets = dat.analyst_price_targets
qrtrly_income = dat.quarterly_income_stmt
history = dat.history(period='1mo')
option_chain = dat.option_chain(dat.options[0]).calls

print("info:")
print("-------------------------------------------")
print(info)
print("calendar:")
print("-------------------------------------------")
print(calendar)
print("price_targets:")
print("-------------------------------------------")
print(price_targets)
print("qrtrly_income:")
print("-------------------------------------------")
print(qrtrly_income)
print("history:")
print("-------------------------------------------")
print(history)
print("option_chain:")
print("-------------------------------------------")
print(option_chain)
