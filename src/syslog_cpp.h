#ifndef __syslogwrapper
#define __syslogwrapper

#include <iostream>
#include <streambuf>
#include <string>
#include <utility>
namespace csyslog {
#include <syslog.h>
}

namespace syslog
{
	struct level {
		enum priority {
			emerge = LOG_EMERG,
			alert  = LOG_ALERT,
			critic = LOG_CRIT,
			error  = LOG_ERR,
			warn   = LOG_WARNING,
			notice = LOG_NOTICE,
			info   = LOG_INFO,
			debug  = LOG_DEBUG
		};
	};
	class streambuf : public std::streambuf
	{
	private:
		std::string buffer;
		int debug_level;
	public:
		streambuf() : debug_level(level::debug)	{ }
		void level(int level) { debug_level = level; }
	protected:
		int sync()
		{
			if (buffer.size()) {
				csyslog::syslog(debug_level, "%s", buffer.c_str());
				buffer.erase();
			}
			return 0;
		}
		int_type overflow(int_type c)
		{
			if(c == traits_type::eof()) sync();
			else buffer += static_cast<char>(c);
			return c;
		}
	};
	class syslog_ostream : public std::ostream
	{
		streambuf log_buffer;
	public:
		syslog_ostream() : std::ostream(&log_buffer) {}
		syslog_ostream& operator<<(const level::priority lev) {
			log_buffer.level(lev);
			return *this;
		}
		void openlog(const char* procname)
		{
			csyslog::openlog( procname, LOG_CONS | LOG_PID | LOG_NDELAY | LOG_LOCAL1, LOG_USER );
		}
		void closelog()
		{
			csyslog::closelog();
		}
		void setlogmask(const level::priority level)
		{
			csyslog::setlogmask(LOG_UPTO(level));
		}
		syslog_ostream& operator << ( std::ostream&(*f)(std::ostream&)) {
			*this << f;
			return *this;
		}
	};
}
extern syslog::syslog_ostream logger;

#endif
